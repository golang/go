package c

//go:embed Static/*
var Static embed.FS //@rename("Static", "static")