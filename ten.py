import random
import math

class HomomorphicEncryption:
    def __init__(self, modulus):
        self.modulus = modulus

    def encrypt(self):
        random_number = random.randint(1, self.modulus - 1)
        encrypted_message = (message * random_number) % self.modulus
        return encrypted_message, random_number

    def decrypt(self, encrypted_message, random_number):
        multiplicative_inverse = pow(random_number, -1, self.modulus)
        decrypted_message = (encrypted_message * multiplicative_inverse) % self.modulus
        return decrypted_message

# Example usage:

modulus = 1024

homomorphic_encryption = HomomorphicEncryption(modulus)

message = 123

encrypted_message, random_number = homomorphic_encryption.encrypt()

print("Encrypted message:", encrypted_message)

decrypted_message = homomorphic_encryption.decrypt(encrypted_message, random_number)

print("Decrypted message:", decrypted_message)