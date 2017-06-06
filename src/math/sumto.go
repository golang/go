package math

/* This is to add number from one to whatever you are adding to.
If you want this explained go to:
http://betterexplained.com/articles/techniques-for-adding-the-numbers-1-to-100/*/

func SumTo(maxint int64) {
  var sum int64
  var subsum1 int64
  var subsum2 int64
  subsum1=maxint+1
  subsum2=subsum1*maxint
  sum=subsum2/2
  return sum
}
