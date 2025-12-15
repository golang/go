package cipher

import (
	"bytes"
	"testing"
)

func TestOTP(t *testing.T) {
	msg := []byte("Hello OTP!")
	ciphertext, key := OTPEncrypt(string(msg))
	decrypted := OTPDecrypt(ciphertext, key)

	if !bytes.Equal([]byte(decrypted), msg) {
		t.Errorf("Decrypted message does not match original")
	}
}
