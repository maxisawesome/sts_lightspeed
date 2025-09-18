//
// Created by keega on 9/24/2021.
//
#include <sstream>
#include <algorithm>

#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "game/Map.h"
#include "game/Neow.h"
#include "game/GameContext.h"

#include "slaythespire.h"

namespace sts {

    NNInterface::NNInterface() :
            cardEncodeMap(createOneHotCardEncodingMap()),
            bossEncodeMap(createBossEncodingMap()) {}

    int NNInterface::getCardIdx(Card c) const {
        int idx = cardEncodeMap[static_cast<int>(c.id)] * 2;
        if (idx == -1) {
            std::cerr << "attemped to get encoding idx for invalid card" << std::endl;
            assert(false);
        }

        if (c.isUpgraded()) {
            idx += 1;
        }

        return idx;
    }

    std::array<int, 20> NNInterface::convertNeowOptionToGenericChoice(const Neow::Option& option) const {
        std::array<int, 20> features{};
        
        // Choice type flags (8 features)
        bool isCardReward = false;
        bool isRelicReward = false;
        bool isGoldReward = false;
        bool isHpChange = false;
        bool isCardUpgrade = false;
        bool isCardRemoval = false;
        bool isCardTransform = false;
        bool isShopAction = false;
        
        // Binary flags (2 features)
        bool isCurse = false;
        bool isUpgrade = false;
        
        // ID fields (2 features)
        int cardId = 0;
        int relicId = 0;
        
        // Reward amounts (5 features, normalized)
        float goldAmount = 0.0f;
        float hpChangeAmount = 0.0f;
        float maxHpChangeAmount = 0.0f;
        float goldCost = 0.0f;
        float hpCost = 0.0f;
        
        // Analyze bonus
        switch (option.r) {
            case Neow::Bonus::THREE_CARDS:
            case Neow::Bonus::ONE_RANDOM_RARE_CARD:
            case Neow::Bonus::THREE_RARE_CARDS:
            case Neow::Bonus::RANDOM_COLORLESS:
            case Neow::Bonus::RANDOM_COLORLESS_2:
                isCardReward = true;
                break;
                
            case Neow::Bonus::RANDOM_COMMON_RELIC:
            case Neow::Bonus::ONE_RARE_RELIC:
            case Neow::Bonus::BOSS_RELIC:
                isRelicReward = true;
                break;
                
            case Neow::Bonus::HUNDRED_GOLD:
                isGoldReward = true;
                goldAmount = 100.0f / 1000.0f; // Normalize by 1000
                break;
                
            case Neow::Bonus::TWO_FIFTY_GOLD:
                isGoldReward = true;
                goldAmount = 250.0f / 1000.0f; // Normalize by 1000
                break;
                
            case Neow::Bonus::TEN_PERCENT_HP_BONUS:
                isHpChange = true;
                maxHpChangeAmount = 0.1f; // 10% as fraction
                break;
                
            case Neow::Bonus::TWENTY_PERCENT_HP_BONUS:
                isHpChange = true;
                maxHpChangeAmount = 0.2f; // 20% as fraction
                break;
                
            case Neow::Bonus::UPGRADE_CARD:
                isCardUpgrade = true;
                break;
                
            case Neow::Bonus::REMOVE_CARD:
            case Neow::Bonus::REMOVE_TWO:
                isCardRemoval = true;
                break;
                
            case Neow::Bonus::TRANSFORM_CARD:
            case Neow::Bonus::TRANSFORM_TWO_CARDS:
                isCardTransform = true;
                break;
                
            default:
                break;
        }
        
        // Analyze drawback
        switch (option.d) {
            case Neow::Drawback::TEN_PERCENT_HP_LOSS:
                isHpChange = true;
                maxHpChangeAmount -= 0.1f; // Subtract 10%
                break;
                
            case Neow::Drawback::NO_GOLD:
                hpCost = 1.0f; // Conceptually costly
                break;
                
            case Neow::Drawback::CURSE:
                isCurse = true;
                break;
                
            case Neow::Drawback::PERCENT_DAMAGE:
                isHpChange = true;
                hpChangeAmount = -0.3f; // -30% HP damage
                break;
                
            case Neow::Drawback::LOSE_STARTER_RELIC:
                hpCost = 1.0f; // Conceptually costly
                break;
                
            default:
                break;
        }
        
        // Pack features into array
        features[0] = isCardReward ? 1 : 0;
        features[1] = isRelicReward ? 1 : 0;
        features[2] = isGoldReward ? 1 : 0;
        features[3] = isHpChange ? 1 : 0;
        features[4] = isCardUpgrade ? 1 : 0;
        features[5] = isCardRemoval ? 1 : 0;
        features[6] = isCardTransform ? 1 : 0;
        features[7] = isShopAction ? 1 : 0;
        features[8] = isCurse ? 1 : 0;
        features[9] = isUpgrade ? 1 : 0;
        features[10] = cardId;
        features[11] = relicId;
        features[12] = static_cast<int>(goldAmount * 1000); // Convert back to int
        features[13] = static_cast<int>(hpChangeAmount * 100); // Scale to int
        features[14] = static_cast<int>(maxHpChangeAmount * 100); // Scale to int
        features[15] = static_cast<int>(goldCost);
        features[16] = static_cast<int>(hpCost * 100); // Scale to int
        features[17] = 0; // padding_1
        features[18] = 0; // padding_2
        features[19] = 0; // padding_3
        
        return features;
    }

    std::array<int,NNInterface::observation_space_size> NNInterface::getObservation(const GameContext &gc) const {
        std::array<int,observation_space_size> ret {};

        int offset = 0;

        ret[offset++] = std::min(gc.curHp, playerHpMax);
        ret[offset++] = std::min(gc.maxHp, playerHpMax);
        ret[offset++] = std::min(gc.gold, playerGoldMax);
        ret[offset++] = gc.floorNum;

        int bossEncodeIdx = offset + bossEncodeMap.at(gc.boss);
        ret[bossEncodeIdx] = 1;
        offset += 10;

        for (auto c : gc.deck.cards) {
            int encodeIdx = offset + getCardIdx(c);
            ret[encodeIdx] = std::min(ret[encodeIdx]+1, cardCountMax);
        }
        offset += 220;

        for (auto r : gc.relics.relics) {
            int encodeIdx = offset + static_cast<int>(r.id);
            ret[encodeIdx] = 1;
        }
        offset += 178;

        // Add current event as one-hot encoding (58 possible events)
        int eventEncodeIdx = offset + static_cast<int>(gc.curEvent);
        ret[eventEncodeIdx] = 1;
        offset += 58;

        // Add generic choice options (4 options * 20 features per option = 80 features)
        for (int i = 0; i < 4; ++i) {
            const auto& option = gc.info.neowRewards[i];
            int choiceOffset = offset;
            
            // Convert Neow option to generic choice features (20 features per choice)
            auto choiceFeatures = convertNeowOptionToGenericChoice(option);
            for (int j = 0; j < 20; ++j) {
                ret[choiceOffset + j] = choiceFeatures[j];
            }
            offset += 20;
        }

        return ret;
    }

    std::array<int,NNInterface::observation_space_size> NNInterface::getObservationMaximums() const {
        std::array<int,observation_space_size> ret {};
        int spaceOffset = 0;

        ret[0] = playerHpMax;
        ret[1] = playerHpMax;
        ret[2] = playerGoldMax;
        ret[3] = 60;
        spaceOffset += 4;

        // Boss encoding (one-hot): max value 1
        std::fill(ret.begin()+spaceOffset, ret.begin()+spaceOffset+10, 1);
        spaceOffset += 10;

        // Card counts: max value cardCountMax
        std::fill(ret.begin()+spaceOffset, ret.begin()+spaceOffset+220, cardCountMax);
        spaceOffset += 220;

        // Relics (binary): max value 1
        std::fill(ret.begin()+spaceOffset, ret.begin()+spaceOffset+178, 1);
        spaceOffset += 178;

        // Events (one-hot): max value 1
        std::fill(ret.begin()+spaceOffset, ret.begin()+spaceOffset+58, 1);
        spaceOffset += 58;

        // Generic choice options (4 options * 20 features): variable max values
        for (int i = 0; i < 4; ++i) {
            // Choice type flags (8 features): max value 1
            std::fill(ret.begin()+spaceOffset, ret.begin()+spaceOffset+8, 1);
            spaceOffset += 8;
            
            // Binary flags (2 features): max value 1  
            std::fill(ret.begin()+spaceOffset, ret.begin()+spaceOffset+2, 1);
            spaceOffset += 2;
            
            // ID fields (2 features): max card/relic IDs
            ret[spaceOffset++] = 371; // Max card ID
            ret[spaceOffset++] = 177; // Max relic ID
            
            // Reward amounts (5 features): normalized ranges
            ret[spaceOffset++] = 250; // gold_amount (max 250)
            ret[spaceOffset++] = 100; // hp_change_amount (±100)
            ret[spaceOffset++] = 100; // max_hp_change_amount (±100)
            ret[spaceOffset++] = 1000; // gold_cost (max 1000)
            ret[spaceOffset++] = 100; // hp_cost (max 100)
            
            // Padding (3 features): unused
            ret[spaceOffset++] = 0;
            ret[spaceOffset++] = 0;
            ret[spaceOffset++] = 0;
        }

        return ret;
    }

    std::vector<int> NNInterface::createOneHotCardEncodingMap() {
        std::vector<CardId> redCards;
        for (int i = static_cast<int>(CardId::INVALID); i <= static_cast<int>(CardId::ZAP); ++i) {
            auto cid = static_cast<CardId>(i);
            auto color = getCardColor(cid);
            if (color == CardColor::RED) {
                redCards.push_back(cid);
            }
        }

        std::vector<CardId> colorlessCards;
        for (int i = 0; i < srcColorlessCardPoolSize; ++i) {
            colorlessCards.push_back(srcColorlessCardPool[i]);
        }
        std::sort(colorlessCards.begin(), colorlessCards.end(), [](auto a, auto b) {
            return std::string(getCardEnumName(a)) < std::string(getCardEnumName(b));
        });

        std::vector<int> encodingMap(372);
        std::fill(encodingMap.begin(), encodingMap.end(), 0);

        int hotEncodingIdx = 0;
        for (auto x : redCards) {
            encodingMap[static_cast<int>(x)] = hotEncodingIdx++;
        }
        for (auto x : colorlessCards) {
            encodingMap[static_cast<int>(x)] = hotEncodingIdx++;
        }

        return encodingMap;
    }

    std::unordered_map<MonsterEncounter, int> NNInterface::createBossEncodingMap() {
        std::unordered_map<MonsterEncounter, int> bossMap;
        bossMap[ME::SLIME_BOSS] = 0;
        bossMap[ME::HEXAGHOST] = 1;
        bossMap[ME::THE_GUARDIAN] = 2;
        bossMap[ME::CHAMP] = 3;
        bossMap[ME::AUTOMATON] = 4;
        bossMap[ME::COLLECTOR] = 5;
        bossMap[ME::TIME_EATER] = 6;
        bossMap[ME::DONU_AND_DECA] = 7;
        bossMap[ME::AWAKENED_ONE] = 8;
        bossMap[ME::THE_HEART] = 9;
        return bossMap;
    }

    NNInterface* NNInterface::getInstance() {
        if (theInstance == nullptr) {
            theInstance = new NNInterface;
        }
        return theInstance;
    }

}

namespace sts::py {

    void play() {
        sts::SimulatorContext ctx;
        sts::ConsoleSimulator sim;
        sim.play(std::cin, std::cout, ctx);
    }

    search::ScumSearchAgent2* getAgent() {
        static search::ScumSearchAgent2 *agent = nullptr;
        if (agent == nullptr) {
            agent = new search::ScumSearchAgent2();
            agent->pauseOnCardReward = true;
        }
        return agent;
    }

    void playout(GameContext &gc) {
        auto agent = getAgent();
        agent->playout(gc);
    }

    std::vector<Card> getCardReward(GameContext &gc) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;

        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return {};
        }

        const auto &r = gc.info.rewardsContainer;
        const auto &cardList = r.cardRewards[r.cardRewardCount-1];
        return std::vector<Card>(cardList.begin(), cardList.end());
    }

    void pickRewardCard(GameContext &gc, Card card) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;
        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return;
        }
        auto &r = gc.info.rewardsContainer;
        gc.deck.obtain(gc, card);
        r.removeCardReward(r.cardRewardCount-1);
    }

    void skipRewardCards(GameContext &gc) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;
        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return;
        }

        if (gc.hasRelic(RelicId::SINGING_BOWL)) {
            gc.playerIncreaseMaxHp(2);
        }

        auto &r = gc.info.rewardsContainer;
        r.removeCardReward(r.cardRewardCount-1);
    }



    // BEGIN MAP THINGS ****************************

    std::vector<int> getNNMapRepresentation(const Map &map) {
        std::vector<int> ret;

        // 7 bits
        // push edges to first row
        for (int x = 0; x < 7; ++x) {
            if (map.getNode(x,0).edgeCount > 0) {
                ret.push_back(true);
            } else {
                ret.push_back(false);
            }
        }

        // for each node in a row, push valid edges to next row, 3 bits per node, 21 bits per row
        // skip 14th row because it is invariant
        // 21 * 13 == 273 bits
        for (int y = 0; y < 14; ++y) {
            for (int x = 0; x < 7; ++x) {

                bool localEdgeValues[3] {false, false, false};
                auto node = map.getNode(x,y);
                for (int i = 0; i < node.edgeCount; ++i) {
                    auto edge = node.edges[i];
                    if (edge < x) {
                        localEdgeValues[0] = true;
                    } else if (edge == x) {
                        localEdgeValues[1] = true;
                    } else {
                        localEdgeValues[2] = true;
                    }
                }
                ret.insert(ret.end(), localEdgeValues, localEdgeValues+3);
            }
        }

        // room types - for each node there are 6 possible rooms,
        // the first row is always monster, the 8th row is always treasure, 14th is always rest
        // this gives 14-3 valid rows == 11
        // 11 * 6 * 7 = 462 bits
        for (int y = 1; y < 14; ++y) {
            if (y == 8) {
                continue;
            }
            for (int x = 0; x < 7; ++x) {
                auto roomType = map.getNode(x,y).room;
                for (int i = 0; i < 6; ++i) {
                    ret.push_back(static_cast<int>(roomType) == i);
                }
            }
        }

        return ret;
    };

    Room getRoomType(const Map &map, int x, int y) {
        if (x < 0 || x > 6 || y < 0 || y > 14) {
            return Room::INVALID;
        }

        return map.getNode(x,y).room;
    }

    bool hasEdge(const Map &map, int x, int y, int x2) {
        if (x == -1) {
            return map.getNode(x2,0).edgeCount > 0;
        }

        if (x < 0 || x > 6 || y < 0 || y > 14) {
            return false;
        }


        auto node = map.getNode(x,y);
        for (int i = 0; i < node.edgeCount; ++i) {
            if (node.edges[i] == x2) {
                return true;
            }
        }
        return false;
    }

}