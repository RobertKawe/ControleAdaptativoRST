##  Controlador RST com Identificação por Mínimos Quadrados (MMQ) Direto

###  Resumo do Projeto

Este projeto implementa um **Controlador RST** acoplado a um algoritmo de **Identificação de Sistemas por Mínimos Quadrados (MMQ) Direto**. O foco é criar uma abordagem de controle adaptativo que una a flexibilidade de sintonia do RST com a alta eficiência da estimação paramétrica MMQ, dispensando etapas intermediárias.

O objetivo final é elevar a performance em tempo real de sistemas dinâmicos, proporcionando **robustez** e **adaptabilidade** a mudanças ambientais ou paramétricas do sistema.

###  Funcionalidades e Vantagens

| Recurso | Descrição |
| :--- | :--- |
| **Controlador RST** | Arquitetura flexível ideal para ajustes de desempenho, permitindo rastreamento de referência e rejeição de distúrbios eficazes. |
| **Identificação MMQ Direto** | Estima os parâmetros do modelo do sistema de forma rápida e eficiente, fundamental para aplicações em tempo real e sistemas embarcados. |
| **Abordagem Adaptativa** | O controle é continuamente realimentado pelos parâmetros identificados, garantindo que o sistema seja robusto e capaz de operar sob condições dinâmicas. |
| **Otimização em Tempo Real** | Baixa complexidade computacional, tornando a solução proposta ideal para implementações em *hardware* com restrições de processamento. |

### Tecnologias Envolvidas

| Ferramenta | Uso no Projeto |
| :--- | :--- |
| **Python** | Linguagem principal de desenvolvimento. Utilizada para a implementação do controlador, MMQ, simulação e visualização de resultados. |
| **Bibliotecas Científicas** | Uso de bibliotecas como **NumPy** e **SciPy** para manipulação numérica e **Matplotlib** para plotagem. |
| **Teoria de Controle** | Controle RST, Modelagem de Sistemas, MMQ, Sistemas Adaptativos. |
