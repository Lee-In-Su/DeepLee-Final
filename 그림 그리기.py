import cv2, random, os, sys
import numpy as np
from copy import deepcopy
from skimage.metrics import mean_squared_error as compare_mse
import multiprocessing as mp

from sklearn.metrics import jaccard_score


import webcam



filepath ='C:/Users/user/Desktop/style-transfer-video-processor/style_ref/test3.png'
filename, ext = os.path.splitext(os.path.basename(filepath))

img = cv2.imread(filepath)
height, width, channels = img.shape

# hyperparameters
n_initial_genes = 50 #첫번째로 생성할 유전자 개수 50
n_population = 50 #한 세대 당 유전자 그룹의 숫자 50
prob_mutation = 0.01 #돌연변이가 발생할 확률 0.01
prob_add = 0.3 #유전자 그룹에 원이 추가될 확률 0.3
prob_remove = 0.2 # 유전자 그룹에 원을 없앨 확률 0.2

min_radius, max_radius = 5, 15 # 원의 크기 #5,15
save_every_n_iter = 150 #100세대 마다 이미지 저장

# Gene
class Gene(): #유전자에 대한 클래스
  def __init__(self):
    self.center = np.array([random.randint(0, width), random.randint(0, height)]) #동그라미의 센터 캔버스 밖으로 안나가게 설정
    self.radius = random.randint(min_radius, max_radius) #동그라미의 반지름
    self.color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]) #동그라미 색깔

  def mutate(self): #돌연변이 생성
    #mutation_size = max(1, int(round(random.gauss(15, 4)))) / 100 #random gauss
    #mutation_size = max(1, int(round(random.uniform(50, 50)))) / 100 #uniform=인자로 받은 두 값사이의 임의의 float숫자를 반환

    mu1 = max(1, int(round(random.uniform(10, 10)))) / 10
    mu2 = mu1 /5
    mu3 = mu2 / 2
    mu4 = min(1, int(round(random.gauss(4, 2)))) / 5
    mu5 = max(1, int(round(random.randrange(0,5,1))))
    mu6 = (mu5+mu4)/mu3
    mutation_size = mu6 /50

    


    # mutation_size = Sequential()
    # mutation_size.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    # mutation_size.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # mutation_size.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    # mutation_size.add(MaxPooling2D(pool_size=(2, 2)))
    # mutation_size.add(Dropout(0.25))
    # mutation_size.add(Flatten())
    # mutation_size.add(Dense(1000, activation='relu'))
    # mutation_size.add(Dropout(0.5))
    # mutation_size.add(Dense(mutation_size, activation='softmax'))

    
    #가우시안 분포를 사용해서 평균이 15고 표준편차가 4인 숫자를 만들고 100으로 나누어서 평균 0.15
    #평균 15%만큼 유전자 변이를 실행시키겠다.

    r = random.uniform(0, 1)
    if r < 0.33: # radius 33%의 확률로 반지름을 변경
      self.radius = np.clip(random.randint( #randint
        int(self.radius * (1 - mutation_size)),
        int(self.radius * (1 + mutation_size))
      ), 1, 100)
    elif r < 0.66: # center 33%의 확률로 동그라미 위치 변경
      self.center = np.array([
        np.clip(random.randint( #randint
          int(self.center[0] * (1 - mutation_size)),
          int(self.center[0] * (1 + mutation_size))),
        0, width),
        np.clip(random.randint( #randint
          int(self.center[1] * (1 - mutation_size)),
          int(self.center[1] * (1 + mutation_size))),
        0, height)
      ])
    else: # color 33%의 확률로 색깔변경
      self.color = np.array([
        np.clip(random.randint(
          int(self.color[0] * (1 - mutation_size)),
          int(self.color[0] * (1 + mutation_size))),
        0, 255),
        np.clip(random.randint(
          int(self.color[1] * (1 - mutation_size)),
          int(self.color[1] * (1 + mutation_size))),
        0, 255),
        np.clip(random.randint(
          int(self.color[2] * (1 - mutation_size)),
          int(self.color[2] * (1 + mutation_size))),
        0, 255)
      ])

# compute fitness
def compute_fitness(genome): #유전자가 환경에 얼마나 잘 적응했는지 확인
  out = np.ones((height, width, channels), dtype=np.uint8) * 255
  
  #이미지 크기만큼의 배열을 255로 다 채워넣음

  for gene in genome:
    cv2.circle(out, center=tuple(gene.center), radius=gene.radius, color=(int(gene.color[0]), int(gene.color[1]), int(gene.color[2])), thickness=-1)
    #원을 그린다. thickness=-1 원 색을 다 칠하므로 이렇게 설정

  # mean squared error
  fitness = 255. / compare_mse(img, out) #두 이미지 차이 원본과 만든 이미지 차를 계산한다. 낮을수록 좋으므로 역수를 취한다.

  return fitness, out

# compute population
def compute_population(g): #유전자를 한꺼번에 돌연변이로 만드는 함수
  genome = deepcopy(g)

  
  # mutation
  if len(genome) < 200: #200보다 작을때는 하나씩 증가
    for gene in genome:
      if random.uniform(0, 1) < prob_mutation:
        gene.mutate()
  else:
    for gene in random.sample(genome, k=int(len(genome) * prob_mutation)):
      gene.mutate()
      #랜덤샘플을 이용해서 랜덤으로 유전자를 뽑아서 돌연변이로 만든다

  # add gene
  if random.uniform(0, 1) < prob_add:
    genome.append(Gene()) #유전자 더하기

  # remove gene
  if len(genome) > 0 and random.uniform(0, 1) < prob_remove:
    genome.remove(random.choice(genome)) #유전자 빼기

  # compute fitness
  new_fitness, new_out = compute_fitness(genome)

  return new_fitness, genome, new_out

# main
if __name__ == '__main__':
  os.makedirs('result', exist_ok=True)

  p = mp.Pool(mp.cpu_count() - 1) #cpu개수 세기

  # 1st gene
  best_genome = [Gene() for _ in range(n_initial_genes)]
  #첫번째 유전자의 성능을 평가
  best_fitness, best_out = compute_fitness(best_genome)

  n_gen = 0

  while True:
    try:
      results = p.map(compute_population, [deepcopy(best_genome)] * n_population)
    except KeyboardInterrupt:
      p.close()
      break #멀티프로세싱을 위해서 설정 50번을 돌면서 새로운 유전자의 피트니스 점수와 새로운 유전자, 우리가 그린그림을 넘겨준다
#병렬처리로 처리
    results.append([best_fitness, best_genome, best_out])

    new_fitnesses, new_genomes, new_outs = zip(*results)

    best_result = sorted(zip(new_fitnesses, new_genomes, new_outs), key=lambda x: x[0], reverse=True)
#피트니스 점수에 따라 내림차순으로 정렬을 한다.
    best_fitness, best_genome, best_out = best_result[0]
#가장 피트니스 점수가 좋은 유전자와 그림이 저장
    # end of generation
    print('Generation #%s, Fitness %s' % (n_gen, best_fitness))
    n_gen += 1

    # visualize
    if n_gen % save_every_n_iter == 0:
      cv2.imwrite('result/%s_%s.png' % (filename, n_gen), best_out)

    cv2.imshow('best out', best_out)
    cv2.imwrite('C:/Users/user/Desktop/style-transfer-video-processor/style_ref/%s.png' % (filename), best_out)
    if cv2.waitKey(1) == ord('q'):
     p.close()
     break

  cv2.imshow('best out', best_out)
  cv2.waitKey(0)
  cv2.imwrite('C:/Users/user/Desktop/style-transfer-video-processor/style_ref/%s.png' % (filename), best_out)
  
  webcam.main()
