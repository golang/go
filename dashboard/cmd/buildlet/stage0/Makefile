buildlet-stage0.windows-amd64: stage0.go
	GOOS=windows GOARCH=amd64 go build -o $@ --tags=stage0
	cat $@ | (cd ../../upload && go run upload.go --public go-builder-data/$@)
