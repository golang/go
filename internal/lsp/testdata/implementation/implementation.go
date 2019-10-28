package implementation

type ImpP struct{} //@ImpP

func (*ImpP) Laugh() {

}

type ImpS struct{} //@ImpS

func (ImpS) Laugh() {

}

type ImpI interface { //@ImpI
	Laugh()
}

type Laugher interface { //@implementations("augher", ImpP),implementations("augher", ImpI),implementations("augher", ImpS),
	Laugh()
}
