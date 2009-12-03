package chanbug
var C chan<- (chan int)
var D chan<- func()
var E func() chan int
var F func() (func())
