#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

// ###==============================###
//
//  Automaticly get L1 chars. by
//  observing performance degradation
//  when memory access causes cache
//  conflicts
//
// ###==============================###

// ###===== CONSTS =====###
constexpr int SIZE = 1 << 23;
constexpr int MIN_STRIDE = 1 << 10;
constexpr int MAX_STRIDE = 1 << 16;
constexpr int MIN_ASSOC = 4;
constexpr int MAX_ASSOC = 32;
constexpr double ASSOC_THR = 1.2;
constexpr int ASSOC_ITER = 20;
constexpr int LINE_SIZE_ITER = 20;
constexpr double LINE_SIZE_THR = 1.2;
constexpr int N = 1 << 20;
// ###=================###

struct BenchmarkResult {
    uint32_t assoc;
    uint32_t cache_size;
    uint32_t line_size;

    BenchmarkResult() : assoc(0), cache_size(0), line_size(0) {
    }

    void reset() {
        assoc = 0;
        cache_size = 0;
        line_size = 0;
    }

    void update(uint32_t assoc, uint32_t cache_size) {
        this->assoc = assoc;
        this->cache_size = cache_size;
    }

    std::pair<uint32_t, uint32_t> get() {
        return std::make_pair(assoc, cache_size);
    }

    uint32_t get_offset() const {
        return cache_size / assoc;
    }

    void print() const {
        std::cout << " - Cache size: " << this->cache_size << std::endl;
        std::cout << " - Associativity: " << this->assoc << std::endl;
        std::cout << " - Line size: " << this->line_size << std::endl;
    }
};

//###===========###
long long grbg = 0;

// common page size
// #pragma clang diagnostic ignored "-Wattribute_not_type_attr"
alignas(8192) uint32_t arr[SIZE];

std::random_device rd;
std::mt19937 g(rd());
//###============###

double measure(int n) {
    int curr = 0;
    int count = N;

    // Warm up
    for (int i = 0; i < n; i++) {
        curr = arr[curr];
    }

    grbg ^= curr;
    curr = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < count; i++) {
        curr = arr[curr];
    }
    auto end = std::chrono::high_resolution_clock::now();

    grbg ^= curr;

    auto res =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if (grbg == 0)
        return double(res + 1) / count;
    else
        return double(res) / count;
}

void chain_to_arr(int spots, int pre_stride) {
    const int stride = pre_stride / sizeof(uint32_t);
    const auto size = static_cast<std::size_t>(spots);

    auto indices = std::views::iota(0) | std::views::take(size);
    std::vector<uint32_t> b;
    std::ranges::copy(indices, std::back_inserter(b));

    std::ranges::shuffle(b, g);

    for (int i = 0; i < spots; i++) {
        arr[b[i % spots] * stride] = b[(i + 1) % spots] * stride;
    }
}

BenchmarkResult eval_associativity() {
    auto benchmark_result = BenchmarkResult();

    std::map<std::pair<uint32_t, uint32_t>, int> m;

    for (int i = 0; i < ASSOC_ITER; i++) {
        int stride_id = 0;

        std::unordered_map<long long, int> size_count;
        std::unordered_map<long long, int> min_assoc;

        for (int stride = MIN_STRIDE; stride < MAX_STRIDE; stride *= 2, stride_id++) {
            double pre_time = -1;
            int pre_spots = 1;

            for (int spots = MIN_ASSOC; spots < MAX_ASSOC; spots += 2) {
                const int real_spots = spots == 0 ? 1 : spots;

                chain_to_arr(real_spots, stride);
                double time = measure(real_spots);
                double k = time / pre_time;

                const int pre_assoc = pre_spots;
                const int pre_cache_size = pre_assoc * stride;

                if (k > ASSOC_THR) {
                    size_count[pre_cache_size]++;
                    min_assoc[pre_cache_size] = pre_assoc;
                }

                pre_time = time;
                pre_spots = real_spots;
            }
        }

        benchmark_result.reset();

        int best_count = -1;
        int best_cache = 0;
        for (const auto &[fst, snd]: size_count) {
            int cache = fst;
            int cnt = snd;
            if (cnt > best_count || (cnt == best_count && cache < best_cache)) {
                best_count = cnt;
                best_cache = cache;
            }
        }
        if (best_count >= 0) {
            benchmark_result.update(min_assoc[best_cache], best_cache);
        }
        m[benchmark_result.get()]++;
    }

    benchmark_result.reset();
    uint32_t max = 0;

    std::ranges::for_each(m, [&](auto p) {
        auto asandls = p.first;
        if (asandls.second > max) {
            max = asandls.second;
            benchmark_result.update(asandls.first, asandls.second);
        }
    });

    return benchmark_result;
}

int chaining_lines(const BenchmarkResult &bench) {
    const uint32_t offset = bench.get_offset();
    const uint32_t indx = offset / bench.line_size;
    const uint32_t spots = indx * bench.assoc * bench.line_size / sizeof(uint32_t);
    std::vector<uint32_t> b(spots);

    for (uint32_t index = 0; index < indx; index++) {
        for (uint32_t tag = 0; tag < bench.assoc; tag++) {
            for (uint32_t el = 0; el < (bench.line_size / sizeof(uint32_t)); el++) {
                const int field_index = index * bench.line_size;
                const int field_tag = (tag + index * bench.assoc) * offset;

                b[el + (tag + index * bench.assoc) * bench.line_size / sizeof(uint32_t)] =
                        (field_index + field_tag) / sizeof(uint32_t) + el;
            }
        }
    }

    std::shuffle(b.begin() + 1, b.end(), g);

    for (uint32_t i = 0; i < spots; i++) {
        arr[b[i]] = b[(i + 1) % spots];
    }

    return spots;
}

void line_size(BenchmarkResult &bench) {
    std::map<int, int> m;

    for (int i = 0; i < LINE_SIZE_ITER; i++) {
        double pre_time = -1;

        for (uint32_t line_size = 8; line_size <= bench.cache_size / bench.assoc; line_size *= 2) {
            bench.line_size = line_size;
            const int spots = chaining_lines(bench);
            const double time = measure(spots);

            if (const double k = pre_time / time; k > LINE_SIZE_THR) {
                m[line_size * sizeof(uint32_t)]++;
                break;
            }

            pre_time = time;
        }
    }

    int max = 0;
    uint32_t correct_line_size = 0;
    std::ranges::for_each(m, [&](auto p) {
        if (p.second > max && p.first != -1) {
            max = p.second;
            correct_line_size = p.first;
        }
    });

    bench.line_size = correct_line_size;
}

int main() {
    auto res = eval_associativity();
    line_size(res);

    res.print();

    return 0;
}
