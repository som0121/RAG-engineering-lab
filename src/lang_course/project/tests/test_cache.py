import time
from app.cache import ResponseCache


def main():
    # Create a cache with a 3-second TTL
    cache = ResponseCache(ttl_seconds=3)

    query = "What is Python?"
    response = "Python is a high-level programming language."

    # 1. Test cache miss
    print("\n--- TEST 1: CACHE MISS ---")

    result = cache.get(query)

    print("Query:", query)
    print("Response:", result)
    print("Expected: None")
    print("Cache Stats:", cache.stats)

    # 2. Store the response
    print("\n--- TEST 2: STORE RESPONSE ---")

    cache.set(query, response)

    print("Response stored successfully!")
    print("Cache Stats:", cache.stats)

    # 3. Test cache hit
    print("\n--- TEST 3: CACHE HIT ---")

    result = cache.get(query)

    print("Query:", query)
    print("Response:", result)
    print("Expected:", response)
    print("Cache Stats:", cache.stats)

    # 4. Test normalized query
    print("\n--- TEST 4: NORMALIZED QUERY ---")

    result = cache.get("  WHAT IS PYTHON?  ")

    print("Response:", result)
    print("Expected:", response)
    print("Cache Stats:", cache.stats)

    # 5. Test a different query
    print("\n--- TEST 5: DIFFERENT QUERY ---")

    result = cache.get("What is Java?")

    print("Response:", result)
    print("Expected: None")
    print("Cache Stats:", cache.stats)

    # 6. Test TTL expiration
    print("\n--- TEST 6: TTL EXPIRATION ---")

    print("Waiting 4 seconds for the cached response to expire...")
    time.sleep(4)

    result = cache.get(query)

    print("Response after expiration:", result)
    print("Expected: None")
    print("Cache Stats:", cache.stats)

    # 7. Store the response again
    print("\n--- TEST 7: STORE AFTER EXPIRATION ---")

    cache.set(query, response)

    result = cache.get(query)

    print("Response:", result)
    print("Expected:", response)
    print("Cache Stats:", cache.stats)

    print("\n--- ALL TESTS COMPLETED ---")


if __name__ == "__main__":
    main()
