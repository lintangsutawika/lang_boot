import random
from functools import partial
from yeval.task import register_task, YevalTask
from yeval.metrics import math_eval

input_fewshot_examples = [
    """\
Example in English:
9.6. Find the minimum value of the expression $(\\sqrt{2(1+\\cos 2 x)}-\\sqrt{36-4 \\sqrt{5}} \\sin x+2) \\cdot(3+2 \\sqrt{10-\\sqrt{5}} \\cos y-\\cos 2 y) \\cdot$ \
If the answer is not an integer, round it to the nearest integer.
##END##
Example in Bahasa Indonesia:
9.6. Cari nilai minimum dari ekspresi $(\\sqrt{2(1+\\cos 2x)} - \\sqrt{36-4\\sqrt{5}} \\sin x + 2) \\cdot (3 + 2\\sqrt{10-\\sqrt{5}} \\cos y - \\cos 2y) \\cdot$ \
Jika jawabannya bukan bilangan bulat, bulatkan ke bilangan bulat terdekat.\
##END##
""",
    """\
Example in English:
18. (3 points) Li Shuang rides a bike at a speed of 320 meters per minute from location $A$ to location $B$. \
On the way, due to a bicycle malfunction, he pushes the bike and walks for 5 minutes to a place 1800 meters from $B$ to repair the bike. \
After 15 minutes, he continues towards $B$ at 1.5 times his original riding speed, and arrives at $B$ 17 minutes later than the expected time. \
What is Li Shuang's walking speed in meters per minute?
##END##
Example in Bahasa Indonesia:
18. (3 poin) Li Shuang mengendarai sepeda dengan kecepatan 320 meter per menit dari lokasi $A$ ke lokasi $B$. \
Di perjalanannya, karena adanya masalah dengan sepedanya, dia memukul sepeda dan berjalan selama 5 menit hingga ke tempat yang berjarak 1800 meter dari $B$ untuk memperbaiki sepedanya. \
Setelah 15 menit, dia melanjutkan perjalanan ke $B$ dengan kecepatan 1,5 kali kecepatan aslinya, dan tiba di $B$ 17 menit lebih lambat dari waktu yang diharapkan. \
Berapa kecepatan berjalan Li Shuang dalam meter per menit?
##END##
""",
    """\
Example in English:
Find all triples $(m,p,q)$ where $ m $ is a positive integer and $ p , q $ are primes.
$$ 2^m p^2 + 1 = q^5 \\]\
##END##
Example in Bahasa Indonesia:
Cari semua triple $(m,p,q)$ di mana $ m $ adalah bilangan bulat positif dan $ p, q $ adalah bilangan prima.
$$ 2^m p^2 + 1 = q^5 \\]\
##END##
""",
    """\
Example in English:
Example 6 The rules of a \"level-up game\" stipulate: On the $n$-th level, a die must be rolled $n$ times. \
If the sum of the points obtained from these $n$ rolls is greater than $2^{n}$, the level is considered passed. Questions:
(1) What is the maximum number of levels a person can pass in this game?
(2) What is the probability that he can pass the first three levels consecutively?
(Note: A die is a uniform cube with points numbered $1,2,3,4,5,6$ on its faces. \
The number of points on the face that lands up after rolling the die is the result of the roll.)
##END##
Example in Bahasa Indonesia:
Contoh 6 Aturan permainan \"level-up\" menyatakan: Pada tingkat ke-$n$, dadu harus dilempar sebanyak $n$ kali. \
Jika jumlah poin yang diperoleh dari $n$ kali lemparan tersebut lebih dari $2^{n}$, maka tingkat tersebut dianggap dilewati. Pertanyaan:
(1) Berapa banyak tingkat maksimum yang dapat dilalui oleh seorang pemain dalam permainan ini?
(2) Berapakah probabilitas bahwa ia dapat melewati tiga tingkat pertama secara berurutan?
(Catatan: Dadu adalah kubus yang setara dengan angka $1,2,3,4,5,6$ tercetak pada masing-masing sisinya. \
Angka pada sisi yang muncul setelah melempar dadu adalah hasil dari lemparan tersebut.)
##END##
"""
]

reasoning_fewshot_examples = [
    """\
Example in English:
2. This problem is equivalent to finding the smallest positive integer $b$, such that the equation $7 b^{2}+7 b+7=x^{4}$, \
(1) has an integer solution for $x$. Since 7 is a prime number, it follows from equation (1) that 7 is a divisor of $x$. \
Therefore, let $x=7 k$, then equation (1) becomes $b^{2}+b+1=7^{3} k^{4}$. The smallest $b$ occurs when $k$ is at its minimum. \
Taking $k=1$, we then have $b^{2}+b+1=343, b^{2}+b-342=0$, which is $(b-18)(b+19)=0$, yielding the positive integer solution $b=18$. \
Thus, we have $(777)_{18}=\\left(7^{4}\\right)_{10}$.
##END##
Example in Bahasa Indonesia:
2. Masalah ini setara dengan mencari bilangan bulat positif terkecil $b$, sedemikian rupa sehingga persamaan $7 b^{2}+7 b+7=x^{4}$ \
(1) memiliki solusi bulat untuk $x$. Karena 7 adalah bilangan prima, dari persamaan (1) dapat disimpulkan bahwa 7 adalah faktor dari $x$. \
Oleh karena itu, misalkan $x=7 k$, maka persamaan (1) menjadi $b^{2}+b+1=7^{3} k^{4}$. Nilai terkecil $ b$ terjadi ketika $ k$ minimal. \
Dengan $k = 1$, maka kita mendapatkan $b^{2}+b+1=343, b^{2}+b-342=0$, yang setara dengan $(b-18)(b+19)=0$, memberikan solusi bulat positif $b=18$. \
Oleh karena itu, kita memiliki $(777)_{18}=\\left(7^{4}\\right)_{10}$.
##END##
""",
#     """\
# English:
# Solution
# In this problem, all lengths are given in meters and areas in square meters.
# a) A piece of rope has length $x$ and another piece of rope has length $10-x$. \
# Since a square has four sides of equal length, one square will have a side length of $\\frac{x}{4}$ and the other square will have a side length of $\\frac{10-x}{4}$.

# The area of a square with side length $\\ell$ is $\\ell^{2}$. Therefore, one square will have an area of $\\left(\\frac{x}{4}\\right)^{2}=\\frac{x^{2}}{16}$, \
# while the other square will have an area of $\\left(\\frac{10-x}{4}\\right)^{2}=\\frac{100-20 x+x^{2}}{16}$.

# b) Let $S(x)$ be the sum of the areas of the two squares. From the previous part, we have

# $$\nS(x)=\\frac{x^{2}}{16}+\\frac{100-20 x+x^{2}}{16}=\\frac{100-20 x+2 x^{2}}{16}=\\frac{1}{8} x^{2}-\\frac{5}{4} x+\\frac{25}{4}\n$$

# which is a quadratic function. The minimum of a function of the form \
    
# $$
# f(x)=a x^{2}+b x+c
# $$

# with $a>0$ is achieved at $x=\\frac{-b}{2 a}$. Thus, the minimum area will be achieved if

# $$
# x=-\\frac{\\left(-\\frac{5}{4}\\right)}{2 \\frac{1}{8}}=5
# $$

# In other words, if the rope is cut exactly in the middle!

# c) From the previous part, we know that to minimize the sum of the areas, it is necessary to cut the rope exactly in the middle. \
# Well, we claim that to minimize the area with nine cuts (i.e., creating ten squares), it is necessary that all pieces of rope be equal. \
# To show this, consider the following argument: if two of the ten pieces of rope were different, it would be possible to reduce the area by cutting the pieces of rope so that these two were equal \
# (we are using the previous part). Therefore, any two pieces of rope must be equal. Hence, all must be equal!
# Bahasa Indonesia:
# Penyelesaian
# Dalam masalah ini, semua panjang diberikan dalam meter dan semua luas dalam satuan meter persegi.
# a) Sebuah potongan tali memiliki panjang $x$ dan potongan tali lainnya memiliki panjang $10-x$. \
# Sejak segiempat memiliki empat sisi dengan panjang yang sama, satu segiempat akan memiliki panjang sisi $\\frac{x}{4}$ dan potongan tali lainnya akan memiliki panjang sisi $\\frac{10-x}{4}$.

# Luas segiempat dengan panjang sisi $\\ell$ adalah $\\ell^{2}$. Jadi, satu segiempat akan memiliki luas $\\left(\\frac{x}{4}\\right)^{2}=\\frac{x^{2}}{16}$, \
# sementara segiempat lainnya akan memiliki luas $\\left(\\frac{10-x}{4}\\right)^{2}=\\frac{100-20 x+x^{2}}{16}$.

# b) Misalkan $S(x)$ adalah jumlah luas dari dua segiempat tersebut. Dari bagian sebelumnya, kita memiliki

# $$\nS(x)=\\frac{x^{2}}{16}+\\frac{100-20 x+x^{2}}{16}=\\frac{100-20 x+2 x^{2}}{16}=\\frac{1}{8} x^{2}-\\frac{5}{4} x+\\frac{25}{4}\n$$
    
# yang merupakan fungsi kuadrat. Minimum dari fungsi berbentuk

# $$
# f(x)=a x^{2}+b x+c
# $$

# dengan $a>0$ dicapai pada $x=\\frac{-b}{2 a}$. Jadi, luas minimum akan dicapai jika

# $$
# x=-\\frac{\\left(-\\frac{5}{4}\\right)}{2 \\frac{1}{8}}=5
# $$

# Artinya, jika tali dipotong tepat di tengah!

# c) Dari bagian sebelumnya, kita tahu bahwa untuk meminimalkan jumlah luas, diperlukan untuk memotong tali tepat di tengah. \
# Kita klaim bahwa untuk meminimalkan luas dengan sembilan potongan (membuat sepuluh segiempat), diperlukan agar semua potongan tali sama. \
# Untuk menunjukkan hal ini, pertimbangkan argumen berikut: jika dua dari sepuluh potongan tali berbeda, maka mungkin untuk meminimalisir luas dengan memotong potongan tali sehingga kedua potongan tersebut sama \
# (kami menggunakan bagian sebelumnya). Oleh karena itu, setiap dua potongan tali harus sama. Jadi, semua harus sama!
# """,
    """\
Example in English:
【Answer】 72
【Analysis】Key point: Skillful area calculation
The area of the square is 196, so the side length is 14. The overlapping area is 1, so the side length is 1; the area of the larger square is 4 times that of the smaller square, so the side length of the larger square is twice that of the smaller square, and the sum of the side lengths of the larger and smaller squares is $14+1=15$.
Therefore, the side length of the smaller square is $15 \\div 3=5$, and the side length of the larger square is $5 \\times 2=10$.\nThe area of the smaller rectangle is $(5-1) \\times(10-1)=36$, so the area of the two smaller rectangles is $36 \\times 2=72\\left(\\mathrm{~cm}^{2}\\right)$.\
##END##
Example in Bahasa Indonesia:
【Jawaban】72
【Analisis】Hal penting: Penghitungan daerah yang cermat
Luas persegi adalah 196, jadi panjang sisi adalah 14. Daerah yang beririsan adalah 1, jadi panjang sisi adalah 1; luas persegi yang lebih besar adalah 4 kali luas persegi yang lebih kecil, jadi panjang sisi persegi yang lebih besar adalah dua kali panjang sisi persegi yang lebih kecil, dan jumlah panjang sisi persegi yang lebih besar dan lebih kecil adalah $14+1=15$.
Jadi, panjang sisi persegi yang lebih kecil adalah $15 \\div 3=5$, dan panjang sisi persegi yang lebih besar adalah $5 \\times 2=10$.\nLuas persegi panjang yang lebih kecil adalah $(5-1) \\times (10-1) = 36$, jadi luas dua persegi panjang yang lebih kecil adalah $36 \\times 2=72\\left(\\mathrm{~cm}^{2}\\right)$.\
##END##
""",
    """\
Example in English:
Father lost four times, so he had to pay his uncle $4 \\cdot 8=32$ crowns.
However, Father won so many times that even after paying these 32 crowns, he still gained 24 crowns. His total winnings were $32+24=56$ crowns, so he won $56: 8=7$ games.
Father won seven times, lost four times, and drew five times, so he played a total of $7+4+5=16$ games with his uncle.
Suggested scoring. 2 points for determining Father's total winnings; 2 points for the number of games Father won; 2 points for the total number of games played.
##END##
Example in Bahasa Indonesia:
Ayah kalah empat kali, jadi dia harus membayar kakeknya $4 \\cdot 8=32$ koin.
Namun, Ayah menang begitu banyak sehingga meskipun membayar 32 koin tersebut, dia masih mendapatkan keuntungan sebanyak 24 koin. Keuntungannya total adalah $32+24=56$ koin, jadi dia menang $56 : 8=7$ permainan.
Ayah menang tujuh kali, kalah empat kali, dan bermain seri lima kali, jadi dia bermain total $7+4+5=16$ permainan dengan pamannya.
Penilaian yang disarankan. 2 poin untuk menentukan total keuntungan Ayah; 2 poin untuk jumlah permainan yang dimenangkan oleh Ayah; 2 poin untuk jumlah total permainan yang dimainkan.
##END##
""",
]

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

@register_task("open_r1_math_220k")
class OpenR1Math220KTask(YevalTask):
    data_path="open-r1/OpenR1-Math-220k"
    data_name="default"
    # preprocessing=shuffle
    input_text=lambda x: x["problem"]
    output_text=lambda x: x["answer"]
    test_split="train"
    evaluation={"score": lambda x,y: -1}

@register_task("open_r1_math_220k_inference_0")
class OpenR1Math220KTask(OpenR1Math220KTask):
    preprocessing=lambda x: partial(shuffle, seed=1000)(x)

@register_task("open_r1_math_220k_inference_1")
class OpenR1Math220KTask(OpenR1Math220KTask):
    preprocessing=lambda x: partial(shuffle, seed=1001)(x)

@register_task("open_r1_math_220k_inference_2")
class OpenR1Math220KTask(OpenR1Math220KTask):
    preprocessing=lambda x: partial(shuffle, seed=1002)(x)

@register_task("open_r1_math_220k_inference_3")
class OpenR1Math220KTask(OpenR1Math220KTask):
    preprocessing=lambda x: partial(shuffle, seed=1003)(x)

@register_task("open_r1_math_220k_inference_4")
class OpenR1Math220KTask(OpenR1Math220KTask):
    preprocessing=lambda x: partial(shuffle, seed=1004)(x)


# @register_task("translate_open_r1_input")
class OpenR1Math220KTask(YevalTask):
    data_path="open-r1/OpenR1-Math-220k"
    data_name="default"
    input_text=lambda x: "\n".join(input_fewshot_examples)+"\nExample in English:\n"+x['problem']+"\n##END##\nExample in Bahasa Indonesia:\n"
    output_text=lambda x: -1
    test_split="train"
    sampling_args={"stop": ["##END##"]}
    evaluation={"score": lambda x,y: -1}

# @register_task("translate_open_r1_reasoning")
class OpenR1220KReasoningTraces(OpenR1Math220KTask):
    input_text=lambda x: "\n".join(reasoning_fewshot_examples)+"\nExample in English:\n"+x['solution']+"\n##END##\nExample in Bahasa Indonesia:\n"
    # input_text=lambda x: x['solution']

if __name__ == "__main__":
    pass
