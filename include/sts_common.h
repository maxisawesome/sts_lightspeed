//
// Created by gamerpuppy on 9/4/2021.
//

#ifndef STS_LIGHTSPEED_STS_COMMON_H
#define STS_LIGHTSPEED_STS_COMMON_H
// #define sts_print_debug
#define sts_asserts

// #define sts_action_queue_use_raw_array
// #define sts_fixed_list_use_raw_array
// #define sts_card_manager_use_fixed_list

#include <cstdint>
#include <unordered_map>

// First define the sts namespace and CardId enum
namespace sts
{
    enum class CardId : uint16_t; // Forward declaration
}

// Then add hash support for CardId enum class
namespace std
{
    template <>
    struct hash<sts::CardId>
    {
        size_t operator()(const sts::CardId &id) const
        {
            return hash<uint16_t>()(static_cast<uint16_t>(id));
        }
    };
}

#endif // STS_LIGHTSPEED_STS_COMMON_H
