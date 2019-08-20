package otp

import (
	"math/rand"
	"strconv"
	"strings"
)

func GenerateOtp(length int)int{
	max,_:=strconv.Atoi(strings.Repeat("9",length))
	min,_:=strconv.Atoi(strings.Repeat("0",length))
	otp:=rand.Intn(max-min)+min
	return otp
}
