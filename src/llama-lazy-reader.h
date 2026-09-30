#pragma once

#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

struct llama_lazy_reader {
#ifdef _WIN32
    // pread()/open() are unavailable on Windows; --lazy-mode on-direct falls
    // back to the lazy mmap reads there (see llama_model_base::load_lazy_reader)
    const int64_t head_dim = 0;

    void gather(const int32_t *, int64_t, float *) const {
        GGML_ABORT("lazy direct reads are not supported on this platform");
    }
#else
    llama_lazy_reader(int fd, size_t base, size_t row_size, int64_t n_rows, int n_threads,
                      enum ggml_type type, int64_t head_dim)
        : fd(fd), base(base), row_size(row_size), n_rows(n_rows), n_threads(n_threads),
          head_dim(head_dim), to_float(type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(type)->to_float) {
        GGML_ASSERT((type == GGML_TYPE_F32 || to_float != nullptr) && head_dim > 0);
    }

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader() {
        {
            std::lock_guard<std::mutex> lk(pm);
            stop = true;
        }
        cv_work.notify_all();
        for (auto & t : pool) {
            t.join();
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }

    const int      fd;
    const size_t   base;       // file offset of row 0
    const size_t   row_size;   // bytes per quantized row
    const int64_t  n_rows;
    const int      n_threads;  // in-flight read workers
    const int64_t  head_dim;
    ggml_to_float_t to_float;  // same dequantizer the ggml_get_rows CPU kernel uses

    // fill dst with the n gathered rows, dequantized to F32:
    // dst[slot * head_dim, ...) = to_float(table[rows[slot]])
    // thread-safe; never lets an exception escape a worker thread
    void gather(const int32_t * rows, int64_t n, float * dst) const {
        std::lock_guard<std::mutex> one(gather_mtx); // the pool runs one gather at a time
        const auto t0 = std::chrono::steady_clock::now();

        std::vector<std::pair<int32_t, int32_t>> pairs;
        pairs.reserve(n);
        for (int64_t i = 0; i < n; ++i) {
            GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < n_rows);
            pairs.emplace_back(rows[i], (int32_t) i);
        }

        std::sort(pairs.begin(), pairs.end()); // equal rows adjacent, file order

        // one work item per distinct row: runs[r] .. runs[r + 1] share a row
        std::vector<int64_t> runs;
        runs.reserve(n + 1);
        for (int64_t i = 0; i < n; ++i) {
            if (i == 0 || pairs[i].first != pairs[i - 1].first) {
                runs.push_back(i);
            }
        }
        runs.push_back(n);
        const int64_t n_runs = (int64_t) runs.size() - 1;

        // Each missed row is a random ~100 B read from a multi-GiB table, i.e. one NVMe round trip (~0.1 ms),
        // so even a decode-sized gather (a few tokens x all heads) must keep many reads in flight: a persistent
        // pool takes one row at a time. Only a gather of one or two rows stays on the calling thread.
        static const bool force_serial = getenv("LLAMA_LAZY_SERIAL") != nullptr; // A/B: the old one-reader behaviour
        if (n_threads <= 1 || n_runs <= 2 || force_serial) {
            std::vector<uint8_t> bounce(row_size);
            for (int64_t r = 0; r < n_runs; ++r) {
                read_run(pairs, runs[r], runs[r + 1], dst, bounce);
            }
        } else {
            if (pool.empty()) {
                pool.reserve(n_threads - 1);
                for (int w = 1; w < n_threads; ++w) {
                    pool.emplace_back([this]() { worker(); });
                }
            }
            job_pairs = &pairs;
            job_runs  = &runs;
            job_dst   = dst;
            job_err   = nullptr;
            next.store(0);
            {
                std::lock_guard<std::mutex> lk(pm);
                pending = (int) pool.size();
                ++gen;
            }
            cv_work.notify_all();
            std::vector<uint8_t> bounce(row_size);
            drain(bounce);
            {
                std::unique_lock<std::mutex> lk(pm);
                cv_done.wait(lk, [&] { return pending == 0; });
            }
            job_pairs = nullptr;
            job_runs  = nullptr;
            job_dst   = nullptr;
            if (job_err) {
                std::rethrow_exception(job_err);
            }
        }

        if (stats_on()) {
            st_calls += 1;
            st_rows  += n;
            st_uniq  += n_runs;
            st_us    += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            if (st_calls % 200 == 0) {
                fprintf(stderr, "LAZYSTATS gathers=%lld rows=%lld uniq=%lld total=%.1f ms (%.3f ms/gather)\n",
                        (long long) st_calls, (long long) st_rows, (long long) st_uniq, st_us / 1000.0, st_us / 1000.0 / st_calls);
            }
        }
    }

    // populate the page cache for the rows a later gather() will read; never writes any output, so a wrong
    // prediction only wastes readahead. Safe to call concurrently with gather().
    void prefetch(const int32_t * rows, int64_t n) const {
        std::vector<int32_t> uniq;
        uniq.reserve(n);
        for (int64_t i = 0; i < n; ++i) {
            if (rows[i] >= 0 && (int64_t) rows[i] < n_rows) { uniq.push_back(rows[i]); }
        }
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        const int n_workers = (int) std::min<int64_t>(n_threads, std::max<int64_t>(1, (int64_t) uniq.size() / 32));
        auto run = [&](int w) {
            const int64_t b = (int64_t) uniq.size() * w / n_workers, e = (int64_t) uniq.size() * (w + 1) / n_workers;
            for (int64_t i = b; i < e; ++i) {
                const size_t off = base + (size_t) uniq[i] * row_size;
                ::posix_fadvise(fd, (off_t) off, (off_t) row_size, POSIX_FADV_WILLNEED);
            }
        };
        std::vector<std::thread> workers;
        try {
            for (int w = 1; w < n_workers; ++w) { workers.emplace_back([&run, w]() { run(w); }); }
        } catch (...) {
            for (auto & t : workers) { t.join(); }
            return;
        }
        run(0);
        for (auto & t : workers) { t.join(); }
    }

private:
    // read the row shared by pairs[begin, end), dequantize it once, copy it to every slot that asked for it
    void read_run(const std::vector<std::pair<int32_t, int32_t>> & pairs, int64_t begin, int64_t end, float * dst,
                  std::vector<uint8_t> & bounce) const {
        const size_t off = base + (size_t) pairs[begin].first * row_size;
        for (size_t done = 0; done < row_size; ) {
            const ssize_t n_read = ::pread(fd, bounce.data() + done, row_size - done, off + done);
            if (n_read < 0 && errno == EINTR) {
                continue;
            }
            if (n_read <= 0) {
                throw std::runtime_error(format("lazy direct read of %zu bytes at file offset %zu failed: %s",
                        row_size, off, n_read == 0 ? "unexpected EOF" : strerror(errno)));
            }
            done += n_read;
        }
        float * first = dst + (size_t) pairs[begin].second * head_dim;
        if (to_float) {
            to_float(bounce.data(), first, head_dim);
        } else {
            memcpy(first, bounce.data(), (size_t) head_dim * sizeof(float));
        }
        for (int64_t k = begin + 1; k < end; ++k) {
            memcpy(dst + (size_t) pairs[k].second * head_dim, first, (size_t) head_dim * sizeof(float));
        }
    }

    // take rows until the job is exhausted; errors are kept, never thrown out of a worker
    void drain(std::vector<uint8_t> & bounce) const {
        const auto & runs = *job_runs;
        for (;;) {
            const int64_t r = next.fetch_add(1);
            if (r + 1 >= (int64_t) runs.size()) {
                break;
            }
            try {
                read_run(*job_pairs, runs[r], runs[r + 1], job_dst, bounce);
            } catch (...) {
                std::lock_guard<std::mutex> lk(err_mtx);
                if (!job_err) {
                    job_err = std::current_exception();
                }
            }
        }
    }

    void worker() const {
        std::vector<uint8_t> bounce(row_size);
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(pm);
                cv_work.wait(lk, [&] { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen = gen;
            }
            drain(bounce);
            {
                std::lock_guard<std::mutex> lk(pm);
                if (--pending == 0) {
                    cv_done.notify_one();
                }
            }
        }
    }

    static bool stats_on() {
        static const bool on = getenv("LLAMA_LAZY_STATS") != nullptr;
        return on;
    }

    // persistent read pool (started on the first multi-row gather)
    mutable std::mutex                       gather_mtx;
    mutable std::mutex                       pm;
    mutable std::condition_variable          cv_work;
    mutable std::condition_variable          cv_done;
    mutable std::vector<std::thread>         pool;
    mutable bool                             stop    = false;
    mutable uint64_t                         gen     = 0;
    mutable int                              pending = 0;
    mutable std::atomic<int64_t>             next{0};
    mutable const std::vector<std::pair<int32_t, int32_t>> * job_pairs = nullptr;
    mutable const std::vector<int64_t> *     job_runs  = nullptr;
    mutable float *                          job_dst   = nullptr;
    mutable std::exception_ptr               job_err;
    mutable std::mutex                       err_mtx;

    mutable int64_t st_calls = 0, st_rows = 0, st_uniq = 0, st_us = 0;
#endif
};
