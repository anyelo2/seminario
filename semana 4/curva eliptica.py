from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import hashlib

# clave Pribada A
priv_A = ec.generate_private_key(ec.SECP256K1())
# clave privda B
priv_B = ec.generate_private_key(ec.SECP256K1())

pub_B = priv_B.public_key()
pub_A = priv_A.public_key()


secreto = priv_A.exchange(ec.ECDH(), pub_B)  
secreto1 = priv_B.exchange(ec.ECDH(), pub_A)  

if (secreto == secreto1):
    print("los secretos son iguales")
    semilla_maestra = hashlib.sha256(secreto).digest() 
print("semilla maestra con hash:  ", semilla_maestra)


def derivar_clave(semilla, contexto):
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=contexto.encode(),
    )
    return hkdf.derive(semilla)

clave_permutacion = derivar_clave(semilla_maestra, "permutacion")
print("k1:  ",clave_permutacion)
clave_sustitutcion   = derivar_clave(semilla_maestra, "sustitucion")
print("k2:  ",clave_sustitutcion)
clave_difucion = derivar_clave(semilla_maestra, "difucion")
print("K3:  ", clave_difucion)


#k1_entero = int.from_bytes(clave_permutacion, byteorder='big')
#print(k1_entero)
