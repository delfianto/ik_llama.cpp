#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
inline size_t parse_moe_cache_mib(const std::string & s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("--moe-cache-mib requires a nonnegative integer");
    unsigned long long n;
    try { n = std::stoull(s); }
    catch (const std::out_of_range &) { throw std::invalid_argument("--moe-cache-mib value is too large"); }
    if (n > std::numeric_limits<size_t>::max() / (1024u*1024u))
        throw std::invalid_argument("--moe-cache-mib byte size overflows size_t");
    return (size_t)n * 1024u*1024u;
}
