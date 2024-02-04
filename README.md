# Stonks

Money printer goes brrrrr

```
if losing_money():
    dont()
```

## Before you commit
Run `./check.sh`, and you should see something like this if all is well. 

```
chris@MacBook-Pro-154 stonks % ./check.sh
===================================== test session starts ======================================
platform darwin -- Python 3.11.4, pytest-8.0.0, pluggy-1.4.0
rootdir: /Users/chris/Code/stonks
plugins: anyio-3.6.2
collected 1 item

tests/test_stonks.py .                                                                   [100%]

====================================== 1 passed in 0.00s =======================================
Success: no issues found in 2 source files
Skipped 2 files
All done! ✨ 🍰 ✨
2 files would be left unchanged.
Success

```

and something like this if something went wrong
```
chris@MacBook-Pro-154 stonks % ./check.sh
===================================== test session starts ======================================
platform darwin -- Python 3.11.4, pytest-8.0.0, pluggy-1.4.0
rootdir: /Users/chris/Code/stonks
plugins: anyio-3.6.2
collected 1 item

tests/test_stonks.py .                                                                   [100%]

====================================== 1 passed in 0.00s =======================================
main.py:2: error: Argument 1 to "add_two_numbers" has incompatible type "str"; expected "int"  [arg-type]
main.py:2: error: Argument 2 to "add_two_numbers" has incompatible type "str"; expected "int"  [arg-type]
Found 2 errors in 1 file (checked 2 source files)
Skipped 2 files
All done! ✨ 🍰 ✨
2 files would be left unchanged.
TESTS/CHECKS FAILED. DO NOT COMMIT.
```

At this point, please fix the error and **DO NOT COMMIT**.   
If you do, I will find you. This is not a threat, this is a promise.   
JK, **or am I?**.


## Q & A
Question: Should I have properly pacakged this as a Python package?  
Answer: Yes

Question: How do you plan on managing Python versions and dependencies?  
Answer: Yesn't

Question: Should this be a pre-commit hook via GitHub actions?   
Answer: Yes.  

Followup question: Am I going to spend the time to do that for this school project?  
Answer: No.  

