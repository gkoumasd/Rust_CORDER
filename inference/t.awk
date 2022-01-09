BEGIN{c=0; s=0}
{
 n=split($1,a,/:/)
 original = tolower(a[n-1])
 predicted = tolower($2)
 print original, predicted
 if (predicted == original) {
         c = c+1
 }
 s = s+1
}
END{
 printf("%f\n", c/s);
}
