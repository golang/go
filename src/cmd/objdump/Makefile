all: x86.go armasm.go

x86.go: bundle
	./bundle -p main -x x86_ rsc.io/x86/x86asm | gofmt >x86.go

armasm.go: bundle
	./bundle -p main -x arm_ rsc.io/arm/armasm | gofmt >armasm.go

bundle:
	go build -o bundle code.google.com/p/rsc/cmd/bundle
