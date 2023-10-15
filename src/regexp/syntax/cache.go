package syntax

import (
	"math"
	"sync"
	"syscall"
	"time"
)

type cacheMap[T any] struct {
	value map[string]T
	err map[string]error
	lastUse map[string]time.Time
	mu sync.Mutex
	null T
}

func newCache[T any]() cacheMap[T] {
	return cacheMap[T]{
		value: map[string]T{},
		err: map[string]error{},
		lastUse: map[string]time.Time{},
	}
}

// get returns a value or an error if it exists
//
// if the object key does not exist, it will return both a nil/zero value (of the relevant type) and nil error
func (cache *cacheMap[T]) get(key string) (T, error) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	if val, ok := cache.value[key]; ok {
		cache.lastUse[key] = time.Now()
		return val, nil
	}else if err, ok := cache.err[key]; ok {
		cache.lastUse[key] = time.Now()
		return cache.null, err
	}

	return cache.null, nil
}

// set sets or adds a new key with either a value, or an error
func (cache *cacheMap[T]) set(key string, value T, err error) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	if err != nil {
		cache.err[key] = err
		delete(cache.value, key)
		cache.lastUse[key] = time.Now()
	}else{
		cache.value[key] = value
		delete(cache.err, key)
		cache.lastUse[key] = time.Now()
	}
}

// delOld removes old cache items
func (cache *cacheMap[T]) delOld(cacheTime time.Duration){
	cache.mu.Lock()
	defer cache.mu.Unlock()

	if cacheTime == 0 {
		for key := range cache.lastUse {
			delete(cache.value, key)
			delete(cache.err, key)
			delete(cache.lastUse, key)
		}
		return
	}

	now := time.Now().UnixNano()

	for key, lastUse := range cache.lastUse {
		if now - lastUse.UnixNano() > int64(cacheTime) {
			delete(cache.value, key)
			delete(cache.err, key)
			delete(cache.lastUse, key)
		}
	}
}

var regexpCache cacheMap[*Regexp] = newCache[*Regexp]()
var progCache cacheMap[*Prog] = newCache[*Prog]()

func init(){
	go func(){
		for {
			time.Sleep(10 * time.Minute)

			// default: remove cache items have not been accessed in over 2 hours
			cacheTime := 2 * time.Hour

			// sysFreeMemory returns the total free system memory in megabytes
			mb := sysFreeMemory()
			if mb < 200 && mb != 0 {
				// low memory: remove cache items have not been accessed in over 10 minutes
				cacheTime = 10 * time.Minute
			}else if mb < 500 && mb != 0 {
				// low memory: remove cache items have not been accessed in over 30 minutes
				cacheTime = 30 * time.Minute
			}else if mb < 2000 && mb != 0 {
				// low memory: remove cache items have not been accessed in over 1 hour
				cacheTime = 1 * time.Hour
			}else if mb > 64000 {
				// high memory: remove cache items have not been accessed in over 12 hour
				cacheTime = 12 * time.Hour
			}else if mb > 32000 {
				// high memory: remove cache items have not been accessed in over 6 hour
				cacheTime = 6 * time.Hour
			}else if mb > 16000 {
				// high memory: remove cache items have not been accessed in over 3 hour
				cacheTime = 3 * time.Hour
			}

			regexpCache.delOld(cacheTime)
			progCache.delOld(cacheTime)

			time.Sleep(10 * time.Second)

			// clear cache if were still critically low on available memory
			if mb := sysFreeMemory(); mb < 10 && mb != 0 {
				regexpCache.delOld(0)
				progCache.delOld(0)
			}
		}
	}()
}

// sysFreeMemory returns the amount of memory available in megabytes
func sysFreeMemory() float64 {
	in := &syscall.Sysinfo_t{}
	err := syscall.Sysinfo(in)
	if err != nil {
		return 0
	}

	// If this is a 32-bit system, then these fields are
	// uint32 instead of uint64.
	// So we always convert to uint64 to match signature.
	return math.Round(float64(uint64(in.Freeram) * uint64(in.Unit)) / 1024 / 1024 * 100) / 100
}
