package burnin

type sendCmdFunc func(string)

func sendCommand(c string) {}

func NewSomething() {
	// This works...
	// var sendCmd sendCmdFunc
	// sendCmd = sendCommand

	// So does this...
	//sendCmd := sendCmdFunc(sendCommand)

	// This fails...
	sendCmd := sendCommand

	_ = sendCmd
}
