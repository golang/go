buildlet-stage0.windows-amd64: stage0.go
	GOOS=windows GOARCH=amd64 go build -o $@ --tags=stage0
	cat $@ | (cd ../../coordinator/buildongce && go run create.go --write_object=go-builder-data/$@)
