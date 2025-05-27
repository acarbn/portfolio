import matplotlib.pyplot as plt
import numpy as np
from joblib import load

def plot_decision_boundary(kernel,model, X, y):
  plt.figure(figsize=(6, 6))
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=60, edgecolors='k', alpha=0.7)
  ax = plt.gca()
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()

  xx = np.linspace(xlim[0], xlim[1], 30)
  yy = np.linspace(ylim[0], ylim[1], 30)
  YY, XX = np.meshgrid(yy, xx)
  xy = np.vstack([XX.ravel(), YY.ravel()]).T
  Z = model.decision_function(xy).reshape(XX.shape)

  ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
    linestyles=['--', '-', '--'])

  ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
        s=150, linewidth=1.5, facecolors='none', edgecolors='k')

  plt.title("Job Decision - Faker Data")
  plt.xlabel("Years of Programming Experience (standardized)")
  plt.ylabel("Technical Score (standardized)")
  plt.grid(True)
  plt.axis('equal')
  plt.savefig("singleHW/"+kernel + ".png", dpi=300)
  plt.show()

def predict_outcome(example):
    model=load('singleHW/linearSVCmodel.joblib')
    scaler = load('singleHW/scaler.pkl')
    example_scaler=scaler.transform(example)
    outcome=model.predict(example_scaler)
    return outcome

    
