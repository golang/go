package cipher

import "crypto/rand"

func KeyGen(message string) []byte {
	key := make([]byte, len(message))

	_, err := rand.Read(key)
	if err != nil {
		panic(err)
	}
	return key
}

func OTPEncrypt(message string) (encrypted, key []byte) {
	encrypted = make([]byte, len(message))
	key = KeyGen(message)
	for i := 0; i < len(message); i++ {
		encrypted[i] = message[i] ^ key[i]
	}
	return
}

func OTPDecrypt(encrypted, key []byte) []byte {
	decrypted := make([]byte, len(encrypted))
	for i := 0; i < len(encrypted); i++ {
		decrypted[i] = encrypted[i] ^ key[i]
	}
	return decrypted
}
