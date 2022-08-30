package interfaces

type first interface {
	common() //@mark(firCommon, "common"),refs("common", firCommon, xCommon, zCommon)
	firstMethod() //@mark(firMethod, "firstMethod"),refs("firstMethod", firMethod, xfMethod, zfMethod)
}

type second interface {
	common() //@mark(secCommon, "common"),refs("common", secCommon, yCommon, zCommon)
	secondMethod() //@mark(secMethod, "secondMethod"),refs("secondMethod", secMethod, ysMethod, zsMethod)
}

type s struct {}

func (*s) common() {} //@mark(sCommon, "common"),refs("common", sCommon, xCommon, yCommon, zCommon)

func (*s) firstMethod() {} //@mark(sfMethod, "firstMethod"),refs("firstMethod", sfMethod, xfMethod, zfMethod)

func (*s) secondMethod() {} //@mark(ssMethod, "secondMethod"),refs("secondMethod", ssMethod, ysMethod, zsMethod)

func main() {
	var x first = &s{}
	var y second = &s{}

	x.common() //@mark(xCommon, "common"),refs("common", firCommon, xCommon, zCommon)
	x.firstMethod() //@mark(xfMethod, "firstMethod"),refs("firstMethod", firMethod, xfMethod, zfMethod)
	y.common() //@mark(yCommon, "common"),refs("common", secCommon, yCommon, zCommon)
	y.secondMethod() //@mark(ysMethod, "secondMethod"),refs("secondMethod", secMethod, ysMethod, zsMethod)

	var z *s = &s{}
	z.firstMethod() //@mark(zfMethod, "firstMethod"),refs("firstMethod", sfMethod, xfMethod, zfMethod)
	z.secondMethod() //@mark(zsMethod, "secondMethod"),refs("secondMethod", ssMethod, ysMethod, zsMethod)
	z.common() //@mark(zCommon, "common"),refs("common", sCommon, xCommon, yCommon, zCommon)
}
