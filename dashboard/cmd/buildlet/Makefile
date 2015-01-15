buildlet: buildlet.go
	go build --tags=extdep

buildlet.linux-amd64: buildlet.go
	GOOS=linux GOARCH=amd64 go build -o $@ --tags=extdep
	cat $@ | (cd ../upload && go run upload.go --public go-builder-data/$@)

buildlet.openbsd-amd64: buildlet.go
	GOOS=openbsd GOARCH=amd64 go build -o $@ --tags=extdep
	cat $@ | (cd ../upload && go run upload.go --public go-builder-data/$@)

buildlet.plan9-386: buildlet.go
	GOOS=plan9 GOARCH=386 go build -o $@ --tags=extdep
	cat $@ | (cd ../upload && go run upload.go --public go-builder-data/$@)

buildlet.windows-amd64: buildlet.go
	GOOS=windows GOARCH=amd64 go build -o $@ --tags=extdep
	cat $@ | (cd ../upload && go run upload.go --public go-builder-data/$@)

buildlet.darwin-amd64: buildlet.go
	GOOS=darwin GOARCH=amd64 go build -o $@ --tags=extdep
	cat $@ | (cd ../upload && go run upload.go --public go-builder-data/$@)

buildlet.netbsd-amd64: buildlet.go
	GOOS=netbsd GOARCH=amd64 go build -o $@ --tags=extdep
	cat $@ | (cd ../upload && go run upload.go --public go-builder-data/$@)
