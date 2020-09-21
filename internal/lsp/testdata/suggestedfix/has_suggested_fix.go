package suggestedfix

import (
	"log"
)

func goodbye() {
	s := "hiiiiiii"
	s = s //@suggestedfix("s = s", "quickfix")
	log.Print(s)
}
