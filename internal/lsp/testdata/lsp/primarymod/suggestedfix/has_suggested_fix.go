package suggestedfix

import (
	"log"
)

func goodbye() {
	s := "hiiiiiii"
	s = s //@suggestedfix("s = s")
	log.Printf(s)
}
