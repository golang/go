package syntax

import (
	"math"
	"time"

	"github.com/alphadose/haxmap"
	"github.com/pbnjay/memory"
)

type cacheRegexpItem struct {
	regexp *Regexp
	err error
	lastUse time.Time
}

type cacheProgItem struct {
	prog *Prog
	err error
	lastUse time.Time
}

var regexpCache *haxmap.Map[string, cacheRegexpItem] = haxmap.New[string, cacheRegexpItem]()
var progCache *haxmap.Map[string, cacheProgItem] = haxmap.New[string, cacheProgItem]()

func init(){
	go func(){
		for {
			time.Sleep(10 * time.Minute)

			now := time.Now().UnixNano()

			// default: remove cache items have not been accessed in over 2 hours
			cacheTime := int64(2 * time.Hour)

			// memory.FreeMemory() returns the total free system memory in bytes
			// the math below, converts bytes to megabytes
			mb := math.Round(float64(memory.FreeMemory()) / 1024 / 1024 * 100) / 100
			if mb < 200 && mb != 0 {
				// low memory: remove cache items have not been accessed in over 10 minutes
				cacheTime = int64(10 * time.Minute)
			}else if mb < 500 && mb != 0 {
				// low memory: remove cache items have not been accessed in over 30 minutes
				cacheTime = int64(30 * time.Minute)
			}else if mb < 2000 && mb != 0 {
				// low memory: remove cache items have not been accessed in over 1 hour
				cacheTime = int64(1 * time.Hour)
			}else if mb > 64000 {
				// high memory: remove cache items have not been accessed in over 12 hour
				cacheTime = int64(12 * time.Hour)
			}else if mb > 32000 {
				// high memory: remove cache items have not been accessed in over 6 hour
				cacheTime = int64(6 * time.Hour)
			}else if mb > 16000 {
				// high memory: remove cache items have not been accessed in over 3 hour
				cacheTime = int64(3 * time.Hour)
			}

			regexpCache.ForEach(func(key string, val cacheRegexpItem) bool {
				if now - val.lastUse.UnixNano() > cacheTime {
					regexpCache.Del(key)
				}
				return true
			})

			progCache.ForEach(func(key string, val cacheProgItem) bool {
				if now - val.lastUse.UnixNano() > cacheTime {
					progCache.Del(key)
				}
				return true
			})

			time.Sleep(10 * time.Second)

			// clear cache if were still critically low on available memory
			if mb := math.Round(float64(memory.FreeMemory()) / 1024 / 1024 * 100) / 100; mb < 10 && mb != 0 {
				regexpCache.ForEach(func(key string, val cacheRegexpItem) bool {
					return true
				})
	
				progCache.ForEach(func(key string, val cacheProgItem) bool {
					return true
				})
			}
		}
	}()
}
