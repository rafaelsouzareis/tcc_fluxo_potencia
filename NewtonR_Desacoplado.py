import cmath
from typing import Tuple, List, Optional
import sympy
import time

class fluxo_potencia_newton_raphson_desacoplado():

    def __init__(self, impedancia_linhas: list, barras: list, qnt_barras: int) -> None:

        self.impedancia_linhas: list = impedancia_linhas
        self.barras: list = barras
        self.qnt_barras: int = qnt_barras
        self.infinito: int = 1e300
        self.indices_slack: list = [i for i, barra in enumerate(barras) if barra[1] == 'Vθ']
        self.indices_pv: list = [i for i, barra in enumerate(barras) if barra[1] == 'PV']
        
        #self.solucao_newton_raphson_desacoplado(tolerancia=1e-4, num_maximo_iteracoes=20)
            
    def solucao_newton_raphson_desacoplado(self, tolerancia: int, num_maximo_iteracoes: int):
        """
        Soluciona o fluxo de potência utilizando o método de Newton-Raphson Desacoplado.
        """

        dados_execucao = {}
        tempo_execucao_inicial = time.time()
        
        iteracao = 0
        self.__calculo_matriz_admitancia__()
        self.__configurar_condicoes_inicias__()
        delta_p, p_c = self.__fluxo_de_potencia_p__()
        delta_q, p_q = self.__fluxo_de_potencia_q__()
        erro_p = self.__calculo_erro_p__(delta_p)
        erro_q = self.__calculo_erro_p__(delta_p)

        while (iteracao < num_maximo_iteracoes):
            
            #Meia Interação
            delta_p, p_c = self.__fluxo_de_potencia_p__()
            erro_p = self.__calculo_erro_p__(delta_p)
           
            if(erro_p < tolerancia and erro_q < tolerancia):
                break

            H = self.__matriz_jacobiana_H__(p_c, p_q)

            #Acrescentar número infinito nos indices das barras Vθ em H
            for i in self.indices_slack:
                H[i][i] = self.infinito
                
            delta_p_linearizado = [
                delta_p[i]/self.tensoes[i] 
                for i in range(self.qnt_barras)
            ]

            H_inversa = self.__matriz_inversa_sympy__(H)
            vetor_correcao_angulos = self.__vetor_correcao__(H_inversa, delta_p_linearizado)
        
            for i in range(self.qnt_barras):
                self.angulos[i] += vetor_correcao_angulos[i]

            #Meia Iteração
            delta_q, q_c = self.__fluxo_de_potencia_q__()
            erro_q = self.__calculo_erro_q__(delta_q)
        
            if(erro_p < tolerancia and erro_q < tolerancia):
                break       

            L = self.__matriz_jacobiana_H__(p_c, q_c)

            #Acrescentar número infinito nos indices das barras Vθ e PV em L
            for i in self.indices_slack + self.indices_pv:
                L[i][i] = self.infinito
            
            delta_q_linearizado = [
                delta_q[i]/self.tensoes[i] 
                for i in range(self.qnt_barras)
            ]

            L_inversa = self.__matriz_inversa_sympy__(L)
            vetor_correcao_tensoes = self.__vetor_correcao__(L_inversa, delta_q_linearizado)

            for i in range(self.qnt_barras):
                self.tensoes[i] += vetor_correcao_tensoes[i]

            dados_execucao[str(iteracao)] = { 
                    "tensoes_iniciais": [self.tensoes[i] - vetor_correcao_tensoes[i] for i in range(self.qnt_barras)],
                    "angulos_iniciais": [self.angulos[i] - vetor_correcao_angulos[i] for i in range(self.qnt_barras)],
                    "p_calculado": p_c,
                    "q_calculado": q_c,
                    "delta_p": delta_p,
                    "delta_q": delta_q,
                    "jacobiana": [H, L],
                    "jacobiana_inversa": [H_inversa, L_inversa],
                    "vetor_correcao_angulos":  vetor_correcao_angulos,
                    "vetor_correcao_tensoes":  vetor_correcao_tensoes,
                    "tensoes_finais": self.tensoes,
                    "angulos_finais": self.angulos
            }

            iteracao += 1
            
        tempo_execucao_final = time.time() - tempo_execucao_inicial 
        dados_execucao['tempo_execucao'] = tempo_execucao_final
        dados_execucao['matriz_admitancia'] = self.matriz_admitancia

        if(iteracao > num_maximo_iteracoes):
            dados_execucao['convergencia'] = False
        else:
            dados_execucao['convergencia'] = True
            #Valores Finais
            delta_q, q_c = self.__fluxo_de_potencia_q__()
            delta_p, p_c = self.__fluxo_de_potencia_p__()

            dados_execucao['solucao'] = {
                'tensoes': self.tensoes,
                'angulos': self.angulos,
                'p': p_c,
                'q': q_c,
                'tempo': tempo_execucao_final,
                'iteracoes': iteracao
            }

        return dados_execucao
   
    def __vetor_correcao__(self, jacobiana: list, vector_mismatch: list) -> Tuple[List[float]]:
        """
        Multiplica a matriz Jacobiana pelo vetor de mismatches.
        """
        resultado = [0] * len(jacobiana)

        for i in range(len(jacobiana)):
            for j in range(len(vector_mismatch)):
                resultado[i] += jacobiana[i][j] * vector_mismatch[j]
        
        return resultado

    '''
    def __matriz_inversa_sympy__(self, matriz: List[List[float]]) -> Tuple[List[float]]:
        import numpy as np
        matriz_numpy = np.array(matriz)
        matriz_numpy_invertida = np.linalg.inv(matriz_numpy)
        
        return matriz_numpy_invertida.tolist()
    
    '''
    def __matriz_inversa_sympy__(self, matriz: list) -> Tuple[List[float]]:
        """
        Calcula a inversa de uma matriz usando SymPy.
        """
        #matriz_sympy = sympy.Matrix(matriz)
        matriz_sympy_invertida = invert_matrix(matriz)#matriz_sympy.inv()
        return matriz_sympy_invertida
        #return matriz_sympy_invertida.tolist()
          
    def __calculo_matriz_admitancia__(self) -> None:
        """
        Calcula a matriz de admitância do sistema.
        """

        self.matriz_admitancia: list = [
            [complex(0, 0)for _ in range(self.qnt_barras)] 
            for _ in range(self.qnt_barras)
        ]

        for linha in self.impedancia_linhas:
            barra_i, barra_j = linha[0] - 1, linha[1] - 1
            y = 1/complex(linha[2], linha[3])
            b_shunt = complex(0, linha[4] / 2)
            self.matriz_admitancia[barra_i][barra_i] += y + b_shunt
            self.matriz_admitancia[barra_j][barra_j] += y + b_shunt
            self.matriz_admitancia[barra_i][barra_j] -= y
            self.matriz_admitancia[barra_j][barra_i] -= y

    def __configurar_condicoes_inicias__(self) -> None:
        """
        Configura as condições iniciais do sistema.
        """

        self.p_spec, self.q_spec, self.tensoes, self.angulos = [], [], [], []

        for barra in self.barras:

            self.tensoes.append(barra[2] if barra[2] != '-' else 1)
            self.angulos.append(barra[3] if barra[3] != '-' else 0)

            p_g = barra[4] if barra[4] != '-' else 0
            q_g = barra[5] if barra[5] != '-' else 0
            p_l = barra[6] if barra[6] != '-' else 0
            q_l = barra[7] if barra[7] != '-' else 0

            self.p_spec.append(p_g - p_l)
            self.q_spec.append(q_g - q_l)

    def __fluxo_de_potencia_p__(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Calcula o fluxo de potência ativa no sistema.
        """

        V = [
            cmath.rect(self.tensoes[i], self.angulos[i]) 
            for i in range (self.qnt_barras)
        ]

        p_c = [0]*self.qnt_barras
       
        for k in range(self.qnt_barras):
            for m in range(self.qnt_barras):
                p_c[k] += (
                    abs(V[k]) 
                    * abs(V[m]) 
                    * (self.matriz_admitancia[k][m].real 
                    * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m])) 
                    + self.matriz_admitancia[k][m].imag 
                    * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m]))
                    ).real
                )
               
        delta_p = [
            (self.p_spec[i] - p_c[i])/abs(V[k]) 
            for i in range(self.qnt_barras)
        ]
     
        return delta_p, p_c
    
    def __fluxo_de_potencia_q__(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Calcula o fluxo de potência reativa no sistema.
        """
        V = [
            cmath.rect(self.tensoes[i], self.angulos[i]) 
            for i in range (self.qnt_barras)
        ]

        q_c = [0]*self.qnt_barras

        for k in range(self.qnt_barras):
            for m in range(self.qnt_barras):
                q_c[k] += (
                    abs(V[k]) 
                    * abs(V[m]) 
                    * (self.matriz_admitancia[k][m].real 
                    * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m])) 
                    - self.matriz_admitancia[k][m].imag 
                    * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m]))
                    ).real
                )

        delta_q = [
            (self.q_spec[i] - q_c[i])/abs(V[k]) 
            for i in range(self.qnt_barras)
        ]

        return delta_q, q_c

    def __calculo_erro_p__(self, delta_p) -> Optional[float]:
        """
        Calcula o maior erro nos no vetor delta_P.
        """
        p_filtrada = []
    
        #Filtrar delta_p excluindo as barras de tipo "Vθ" 
        for i, barra in enumerate(self.barras):
            if barra[1] not in ["Vθ"]:
                p_filtrada.append(abs(delta_p[i]))

        # Calcular o valor absoluto máximo do vetor concatenado
        if p_filtrada:
            maior_erro = abs(p_filtrada[0])
            for erro in p_filtrada[1:]:
                erro_abs = abs(erro)
                if erro_abs > maior_erro:
                    maior_erro = erro_abs
        else:
            maior_erro = 0

        return maior_erro

    def __calculo_erro_q__(self, delta_q) -> Optional[float]:
        """
        Calcula o maior erro nos no vetor delta_Q.
        """
        q_filtrada = []
    
        # Filtrar delta_q excluindo as barras de tipo "Vθ" e "PV"
        for i, barra in enumerate(self.barras):
            if barra[1] not in ["Vθ", "PV"]:
                q_filtrada.append(abs(delta_q[i]))

        # Calcular o valor absoluto máximo do vetor concatenado
        if q_filtrada:
            maior_erro = abs(q_filtrada[0])
            for erro in q_filtrada[1:]:
                erro_abs = abs(erro)
                if erro_abs > maior_erro:
                    maior_erro = erro_abs
        else:
            maior_erro = 0

        return maior_erro

    def __inicializar_matriz__(self) -> List[int]:
        "Criar matriz qnt_barras x qnt_barras"
        return [
            [0 for _ in range(self.qnt_barras)] 
            for _ in range(self.qnt_barras)
        ]

    def __matriz_jacobiana_H__(self, p:list, q:list) -> Tuple[List[float]]:
        """
        Calcula a submatriz H, matriz Jacobiana.
        """

        V = [
            cmath.rect(self.tensoes[i], self.angulos[i]) 
            for i in range (self.qnt_barras)
        ]
        H = self.__inicializar_matriz__()

        for k in range(self.qnt_barras):
            for m in range(self.qnt_barras):
                if k == m:
                    H[k][k] = (
                        -q[k]/abs(V[k]) 
                        - self.matriz_admitancia[k][k].imag 
                        * abs(V[k])
                    )
                else:
                    H[k][m] += (
                        abs(V[m]) 
                        * (self.matriz_admitancia[k][m].real 
                        * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m])) 
                        - self.matriz_admitancia[k][m].imag 
                        * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m]))
                        ).real
                    )

        return H
    
    def __matriz_jacobiana_L__(self, p:list, q:list) -> Tuple[List[float], List[float]]:
        """
        Calcula a submatriz L, matriz Jacobiana.
        """

        V = [
            cmath.rect(self.tensoes[i], self.angulos[i]) 
            for i in range (self.qnt_barras)
        ]
        L = self.__inicializar_matriz__()

        for k in range(self.qnt_barras):
            for m in range(self.qnt_barras):
                if k == m:
                    L[k][k] =  (
                        q[k]/abs(V[k])**2 
                        - self.matriz_admitancia[k][k].imag
                    )
                else:
                    L[k][m] += (
                        (self.matriz_admitancia[k][m].real 
                        * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m])) 
                        - self.matriz_admitancia[k][m].imag 
                        * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m]))
                        ).real
                    )

        return L
    


def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            sum_u = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum_u
        for j in range(i + 1, n):
            sum_l = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - sum_l) / U[i][i]

    return L, U

def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        sum_y = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - sum_y
    return y

def backward_substitution(U, y):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_x = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_x) / U[i][i]
    return x

def invert_matrix(A):
    n = len(A)
    L, U = lu_decomposition(A)
    I = [[0.0] * n for _ in range(n)]
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        y = forward_substitution(L, e)
        x = backward_substitution(U, y)
        for j in range(n):
            I[j][i] = x[j]
    return I