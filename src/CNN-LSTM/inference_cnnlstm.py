def get_punc(text, splitted=False):
  if not splitted:
    text = text.split()
  x_prepared = dataload_func.get_embedding(text)

  X = torch.from_numpy(x_prepared).float()
  X = X.reshape(1, 300, 10)
  out = model(X)
  out = out.detach().numpy()
  new_outputs = np.argmax(out,axis=2)

  result = []
  for o, t in zip(new_outputs[0], text):
    result.append((t, id2tag[o]))

  return result


text = "من در ایران زندگی میکنم ولی شما چطور زندگی میکنید"
print(get_punc(text))
