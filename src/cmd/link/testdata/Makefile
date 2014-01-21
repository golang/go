all: hello.6 pclntab.6

hello.6: hello.s
	go tool 6a hello.s

pclntab.6: pclntab.s
	go tool 6a pclntab.s

pclntab.s: genpcln.go
	go run genpcln.go >pclntab.s
