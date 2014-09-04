coordinator: main.go
	GOOS=linux go build -o coordinator .

upload: coordinator
	cat coordinator | (cd buildongce && go run create.go --write_object=go-builder-data/coordinator)

