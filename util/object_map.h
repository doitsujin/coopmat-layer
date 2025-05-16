#pragma once

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "rwlock.h"

namespace util {

/**
 * \brief Thread-safe key-value map
 *
 * \tparam K Key type
 * \tparam V Value type. Will be stored
 *    and returned as a shared_ptr.
 */
template<typename K, typename V>
class ObjectMap {

public:

  /**
   * \brief Retrieves item with a given key
   *
   * \param [in] key Key to look up
   * \returns Pointer to the item with the given key,
   *    or \c nullptr if no item with the given key exists
   */
  std::shared_ptr<V> find(const K& key) const {
    std::shared_lock lock(m_lock);

    auto entry = m_map.find(key);

    if (entry == m_map.end())
      return nullptr;

    return entry->second;
  }

  /**
   * \brief Creates item with a given key and arguments
   *
   * \param [in] key Key
   * \param [in] args Constructor arguments
   * \returns Pointer to the created item, or \c nullptr
   *    if an item for the given key already exists.
   */
  template<typename... Args>
  std::shared_ptr<V> create(const K& key, Args&&... args) {
    std::unique_lock lock(m_lock);

    auto result = m_map.emplace(std::piecewise_construct,
      std::tuple(key),
      std::tuple());

    if (!result.second)
      return nullptr;

    result.first->second = std::make_unique<V>(std::forward<Args>(args)...);
    return result.first->second;
  }

  /**
   * \brief Removes item with a given key
   * \param [in] key Key to remove
   */
  void erase(const K& key) {
    std::unique_lock lock(m_lock);
    m_map.erase(key);
  }

private:

  mutable RwLock                            m_lock;
  std::unordered_map<K, std::shared_ptr<V>> m_map;

};

}
