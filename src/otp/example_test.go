package otp_test

import (
	"fmt"
	"otp"
)

func ExampleGenerateOtp() {
	num:=otp.GenerateOtp(6)
	fmt.Println(num)

	//OUTPUT: 434543
}
