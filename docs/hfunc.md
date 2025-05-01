---
title: GARCH heteroskedasticity function syntax guide
nav_order: 4
has_toc: true
---
The option `pt.options.h_func` allows you to define your own GARCH heteroskedasticity function. It must be a dict with keys 
 'h', 'h\_e', 'h\_e2', 'h\_z','h\_z2' and 'h\_e\_z', defining the function h(e,z) and 
its derivatives 

$\frac{\partial h}{\partial e}$,  
$\frac{\partial^2 h}{{\partial e}^2}$,
$\frac{\partial h}{\partial z}$
$\frac{\partial^2 h}{{\partial z}^2}$, 
$\frac{\partial^2 h}{\partial e \partial ez}$ 


respectively. 

The function h(e,z) must be a string with the variable e and optionally z. 
Hence, you define the string of the mathemtical function to be performed, and 
not the function itself. Correspondingly, the derivatives must be strings represeting mathematical expressions, 
and not function defitions. Se 'GARCH heteroskedasticity function syntax guide' in the documentation for information
on how to write the mathematical expressions. \n\n

For example, you could define the function as

```python
import paneltime as pt
pt.options.h_func = {
	'h': 'e^2 + 0.5*e^4 + z',
	'h_e': '2*e + 2*e^3',
	'h_e2': '2 + 6*e^2',
	'h_z': '1',
	'h_z2': '0',
	'h_e_z': '0'
}
``` 


# `exprtk` Expression Syntax Guide

You can define mathematical expressions using this syntax. The expressions will be evaluated efficiently in C++ using the [ExprTk library](https://github.com/ArashPartow/exprtk).

---

## Allowed Variables

Use named variables such as `x`, `y`, `z`.  
You must declare these variables on the Python or C++ side before evaluating the expression.

---

## Supported Operators

| Operator      | Description              | Example           |
|---------------|--------------------------|-------------------|
| `+` `-`       | Addition / Subtraction   | `x + y - 2`       |
| `*` `/` `%`   | Multiplication, Division, Modulo | `x * y / 2` |
| `^`           | Exponentiation           | `x^2`             |
| `==`, `!=`    | Equality / Inequality    | `x == y`          |
| `<` `>` `<=` `>=` | Comparisons         | `x < y ? x : y`   |
| `&&`, `||`, `!` | Logical AND, OR, NOT  | `x > 0 && y > 0`  |

---

## Built-in Functions

| Function      | Description              |
|---------------|--------------------------|
| `abs(x)`      | Absolute value           |
| `sqrt(x)`     | Square root              |
| `log(x)`      | Natural logarithm        |
| `exp(x)`      | Exponential              |
| `sin(x)` `cos(x)` `tan(x)` | Trigonometry |
| `floor(x)` `ceil(x)` | Rounding          |
| `min(x, y)` `max(x, y)` | Minimum / Maximum |
| `sign(x)`     | Returns -1, 0, or +1     |

---

## Conditionals

You can use the ternary operator for branching:

```text
x > y ? x : y
```

This returns `x` if the condition is true, otherwise `y`.

---

## Example Expressions

```text
x^2 + 2*x + 1
log(x) + y^2
x > y ? x - y : y - x
min(x, y) + sqrt(abs(z))
```

---

## Notes

- Do not use Python syntax like `**` for powers — use `^` instead.
- Do not use `math.` prefix for functions (just write `log(x)`, not `math.log(x)`).
- The expression should be a single line and not contain Python `def`, `if`, `for`, etc.


---

## Reference

ExprTk is an open-source C++ math expression parser.  
Documentation: [https://github.com/ArashPartow/exprtk](https://github.com/ArashPartow/exprtk)

