import cmath
from typing import Tuple, List, Optional
import sympy
import time

class fluxo_potencia_newton_raphson():

    def __init__(self, impedancia_linhas: list, barras: list, qnt_barras: int) -> None:

        self.impedancia_linhas: list = impedancia_linhas
        self.barras: list = barras
        self.qnt_barras: int = qnt_barras
        self.infinito: int = 1e300

        self.indices_slack = [i for i, barra in enumerate(barras) if barra[1] == 'Vθ']
        self.indices_pv = [i for i, barra in enumerate(barras) if barra[1] == 'PV']

        #self.solucao_newton_raphson(tolerancia=1e-4, num_maximo_iteracoes=20)

    def solucao_newton_raphson(self, tolerancia: float, num_maximo_iteracoes: int) -> None:
        """
        Soluciona o fluxo de potência utilizando o método de Newton-Raphson.
        """

        dados_execucao = {}
        tempo_execucao_inicial = time.time()

        self.__calculo_matriz_admitancia()
        self.__configurar_condicoes_inicias()

        iteracao = 0

        while (iteracao < num_maximo_iteracoes):
            delta_p, delta_q, p_c, q_c = self.__fluxo_de_potencia()
            erro = self.__calculo_erro(delta_p=delta_p, delta_q=delta_q)

            if(erro >= tolerancia):
                H, N, M, L = self.__matriz_jacobiana_newton_raphson(p_c, q_c)
                
                #Acrescentar número infinito nos indices das barras Vθ em H e Vθ e PV em L
                for i in self.indices_slack:
                    H[i][i] = self.infinito
                for i in self.indices_slack + self.indices_pv:
                    L[i][i] = self.infinito

                matriz_jacobiana_superior = [h + n for h, n in zip(H, N)]
                matriz_jacobiana_inferior = [m + l for m, l in zip(M, L)]
                matriz_jacobiana = matriz_jacobiana_superior + matriz_jacobiana_inferior
                matriz_jacobiana_invertida = self.__matriz_inversa_sympy(matriz_jacobiana)
                vetor_correcao_angulos, vetor_correcao_tensoes = self.__vetor_correcao(matriz_jacobiana_invertida, delta_p + delta_q)

                #Novos angulos e tensoes
                for i in range(self.qnt_barras):
                    self.tensoes[i] += vetor_correcao_tensoes[i]
                    self.angulos[i] += vetor_correcao_angulos[i]

                dados_execucao[str(iteracao)] = { 
                    "tensoes_iniciais": [self.tensoes[i] - vetor_correcao_tensoes[i] for i in range(self.qnt_barras)],
                    "angulos_iniciais": [self.angulos[i] - vetor_correcao_angulos[i] for i in range(self.qnt_barras)],
                    "p_calculado": p_c,
                    "q_calculado": q_c,
                    "delta_p": delta_p,
                    "delta_q": delta_q,
                    "jacobiana": matriz_jacobiana,
                    "jacobiana_inversa": matriz_jacobiana_invertida,
                    "vetor_correcao_angulos":  vetor_correcao_angulos,
                    "vetor_correcao_tensoes":  vetor_correcao_tensoes,
                    "tensoes_finais": self.tensoes,
                    "angulos_finais": self.angulos
                }

                iteracao += 1
            else:
                break

        tempo_execucao_final = time.time() - tempo_execucao_inicial 
        dados_execucao['tempo_execucao'] = tempo_execucao_final
        dados_execucao['matriz_admitancia'] = self.matriz_admitancia

        if(iteracao > num_maximo_iteracoes):
            dados_execucao['convergencia'] = False

        else:
            dados_execucao['convergencia'] = True
            delta_p, delta_q, p_c, q_c = self.__fluxo_de_potencia()
            dados_execucao['solucao'] = {
                'tensoes': self.tensoes,
                'angulos': self.angulos,
                'p': p_c,
                'q': q_c,
                'tempo': tempo_execucao_final,
                'iteracoes': iteracao
            }

        return dados_execucao

    def __vetor_correcao(self, jacobiana: list, vetor_mismatch: list) -> Tuple[List[float]]:
        """
        Multiplica a matriz Jacobiana pelo vetor de mismatches.
        """

        resultado = [0] * len(jacobiana)
        for i in range(len(jacobiana)):
            for j in range(len(vetor_mismatch)):
                resultado[i] += jacobiana[i][j] * vetor_mismatch[j]

        angulos = resultado[:self.qnt_barras]
        tensoes = resultado[self.qnt_barras:2*self.qnt_barras]
        return angulos, tensoes
  
    def __matriz_inversa_sympy(self, matriz: list) -> Tuple[List[float]]:
        """
        Calcula a inversa de uma matriz usando SymPy.
        """

        matriz_sympy = sympy.Matrix(matriz)
        matriz_sympy_invertida = matriz_sympy.inv()
        
        return matriz_sympy_invertida.tolist()
                
    def __calculo_matriz_admitancia(self) -> None:
        """
        Calcula a matriz de admitância do sistema.
        """

        self.matriz_admitancia: list = [
            [complex(0, 0) for _ in range(self.qnt_barras)] 
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

    def __configurar_condicoes_inicias(self) -> None:
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
        
    def __fluxo_de_potencia(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Calcula o fluxo de potência no sistema.
        """

        V = [
            cmath.rect(self.tensoes[i], self.angulos[i]) 
            for i in range (self.qnt_barras)
        ]

        p_c = [0]*self.qnt_barras
        q_c = [0]*self.qnt_barras

        for k in range(self.qnt_barras):
            for m in range(self.qnt_barras):

                p_c[k] += (
                    abs(V[k]) 
                    * abs(V[m]) 
                    * (self.matriz_admitancia[k][m].real 
                    * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m])) 
                    + self.matriz_admitancia[k][m].imag 
                    * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m]))).real
                )
                q_c[k] += (
                    abs(V[k]) 
                    * abs(V[m]) 
                    * (self.matriz_admitancia[k][m].real 
                    * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m])) 
                    - self.matriz_admitancia[k][m].imag 
                    * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m]))).real
                )

        delta_p = [0]*self.qnt_barras
        delta_q = [0]*self.qnt_barras
 
        for i in range(self.qnt_barras):
            delta_p[i] = self.p_spec[i] - p_c[i]
            delta_q[i] = self.q_spec[i] - q_c[i]

        return delta_p, delta_q, p_c, q_c

    def __calculo_erro(self, delta_q, delta_p) -> Optional[float]:
        """
        Calcula o maior erro nos vetores de delta P e delta Q.
        """

        p_filtrada, q_filtrada = [], []
    
        #Filtrar delta_p excluindo as barras de tipo "Vθ" 
        for i, barra in enumerate(self.barras):
            if barra[1] not in ["Vθ"]:
                p_filtrada.append(abs(delta_p[i]))

        # Filtrar delta_q excluindo as barras de tipo "Vθ" e "PV"
        for i, barra in enumerate(self.barras):
            if barra[1] not in ["Vθ", "PV"]:
                q_filtrada.append(abs(delta_q[i]))

        vetor_concatenado = p_filtrada + q_filtrada

        if vetor_concatenado:
            maior_erro = abs(vetor_concatenado[0])
            for erro in vetor_concatenado[1:]:
                erro_abs = abs(erro)
                if erro_abs > maior_erro:
                    maior_erro = erro_abs
        else:
            maior_erro = 0

        return maior_erro

    def __inicializar_matriz(self) -> List[int]:
        "Criar matriz qnt_barras x qnt_barras"
        return [
            [0 for _ in range(self.qnt_barras)] 
            for _ in range(self.qnt_barras)
        ]

    def __matriz_jacobiana_newton_raphson(self, p:list, q:list) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Calcula as submatrizes H, N, M e L da matriz Jacobiana.
        """

        V = [cmath.rect(self.tensoes[i], self.angulos[i]) for i in range (self.qnt_barras)]

        H = self.__inicializar_matriz()
        N = self.__inicializar_matriz()
        M = self.__inicializar_matriz()
        L = self.__inicializar_matriz()

        for k in range(self.qnt_barras):
            for m in range(self.qnt_barras):
                if k == m:
                    H[k][k] = (
                        -q[k] 
                        - self.matriz_admitancia[k][k].imag * abs(V[k])**2
                    )
                    N[k][k] = ((
                        p[k] 
                        + self.matriz_admitancia[k][k].real * abs(V[k])**2) 
                        / abs(V[k])
                    )
                    M[k][k] = (
                        p[k] 
                        - self.matriz_admitancia[k][k].real 
                        * abs(V[k])**2
                    )
                    L[k][k] = (
                        (q[k] 
                         - self.matriz_admitancia[k][k].imag * abs(V[k])**2)
                         *(abs(V[k]))**(-1)
                    )
                else:
                    H[k][m] += (
                        abs(V[k]) 
                        * abs(V[m]) 
                        * (self.matriz_admitancia[k][m].real 
                        * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m])) 
                        - self.matriz_admitancia[k][m].imag 
                        * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m]))).real
                    )
                    N[k][m] += (
                        abs(V[k]) 
                        * (self.matriz_admitancia[k][m].real 
                        * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m])) 
                        + self.matriz_admitancia[k][m].imag 
                        * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m]))).real
                    )
                    M[k][m] += (
                        -abs(V[k]) 
                        * abs(V[m]) 
                        * (self.matriz_admitancia[k][m].real 
                        * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m])) 
                        + self.matriz_admitancia[k][m].imag 
                        * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m]))).real
                    )
                    L[k][m] += (
                        abs(V[k]) 
                        * (self.matriz_admitancia[k][m].real 
                        * cmath.sin(cmath.phase(V[k]) - cmath.phase(V[m])) 
                        - self.matriz_admitancia[k][m].imag 
                        * cmath.cos(cmath.phase(V[k]) - cmath.phase(V[m]))).real
                    )
        
        return H, N, M, L

