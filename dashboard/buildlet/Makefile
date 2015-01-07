buildlet: buildlet.go
	go build --tags=buildlet

buildlet.openbsd-amd64: buildlet.go
	GOOS=openbsd GOARCH=amd64 go build -o $@ --tags=buildlet
	cat $@ | (cd ../coordinator/buildongce && go run create.go --write_object=go-builder-data/$@)

buildlet.plan9-386: buildlet.go
	GOOS=plan9 GOARCH=386 go build -o $@ --tags=buildlet
	cat $@ | (cd ../coordinator/buildongce && go run create.go --write_object=go-builder-data/$@)

buildlet.windows-amd64: buildlet.go
	GOOS=windows GOARCH=amd64 go build -o $@ --tags=buildlet
	cat $@ | (cd ../coordinator/buildongce && go run create.go --write_object=go-builder-data/$@)

buildlet.darwin-amd64: buildlet.go
	GOOS=darwin GOARCH=amd64 go build -o $@ --tags=buildlet
	cat $@ | (cd ../coordinator/buildongce && go run create.go --write_object=go-builder-data/$@)
