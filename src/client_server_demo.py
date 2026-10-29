import time
import shutil
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.metrics import accuracy_score
from concrete.ml.sklearn import LogisticRegression as FheLogReg
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plain_model = SkLogReg(max_iter=1000)
plain_model.fit(X_train, y_train)
t = time.time()
plain_pred = plain_model.predict(X_test)
plain_time = time.time() - t
plain_acc = accuracy_score(y_test, plain_pred)

fhe_model = FheLogReg(n_bits=8)
fhe_model.fit(X_train, y_train)
t = time.time()
fhe_model.compile(X_train)
compile_time = time.time() - t

quant_pred = fhe_model.predict(X_test)
quant_acc = accuracy_score(y_test, quant_pred)

t = time.time()
fhe_pred = fhe_model.predict(X_test, fhe="execute")
fhe_time = time.time() - t
fhe_acc = accuracy_score(y_test, fhe_pred)
agreement = np.mean(fhe_pred == quant_pred)

print("Samples:", len(X_test), "Features:", X.shape[1])
print("Plaintext accuracy:", round(plain_acc, 4))
print("Plaintext time per sample:", plain_time / len(X_test), "s")
print("Quantized clear accuracy:", round(quant_acc, 4))
print("Encrypted accuracy:", round(fhe_acc, 4))
print("Encrypted vs quantized agreement:", round(agreement, 4))
print("Encrypted time per sample:", round(fhe_time / len(X_test), 4), "s")
print("Compile time:", round(compile_time, 2), "s")

shutil.rmtree("deploy", ignore_errors=True)
shutil.rmtree("keys", ignore_errors=True)
FHEModelDev("deploy", fhe_model).save()

client = FHEModelClient("deploy", key_dir="keys")
client.load()
t = time.time()
client.generate_private_and_evaluation_keys(force=True)
keygen_time = time.time() - t
eval_keys = client.get_serialized_evaluation_keys()

server = FHEModelServer("deploy")
server.load()

sample = X_test[:1]
t = time.time()
encrypted_input = client.quantize_encrypt_serialize(sample)
enc_time = time.time() - t
t = time.time()
encrypted_output = server.run(encrypted_input, eval_keys)
server_time = time.time() - t
t = time.time()
result = client.deserialize_decrypt_dequantize(encrypted_output)
dec_time = time.time() - t

print("Key generation time:", round(keygen_time, 2), "s")
print("Evaluation keys size:", round(len(eval_keys) / 1e6, 2), "MB")
print("Plain input size:", sample.nbytes, "bytes")
print("Encrypted input size:", round(len(encrypted_input) / 1e3, 2), "KB")
print("Encrypted output size:", round(len(encrypted_output) / 1e3, 2), "KB")
print("Client encrypt time:", round(enc_time, 4), "s")
print("Server compute time:", round(server_time, 4), "s")
print("Client decrypt time:", round(dec_time, 4), "s")
print("Decrypted result:", result)
