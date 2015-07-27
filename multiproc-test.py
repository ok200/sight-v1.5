import cv2
import 

concurrency = 8    # 8並列とする

# 各プロセスが実行する計算
def subcalc(image):

# 8個のプロセスを用意
pool = mp.Pool(concurrency)

# 各プロセスに subcalc(p) を実行させる
# ここで p = 0,1,...,7
# callbackには各戻り値がlistとして格納される
callback = pool.map(subcalc, range(8))

# 各戻り値の総和を計算
total = sum(callback)

print (total)