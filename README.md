# Projet Convex Optimization — Présentation du Dataset

## *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*

---

## 1. Présentation du dataset

### 1.1 Identité du dataset

**Nom officiel :** Estimation of Obesity Levels Based on Eating Habits and Physical Condition

**Source :** UCI Machine Learning Repository (Dataset ID #544)

**Lien direct :** https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

**Article scientifique d'origine :** Mendoza Palechor, F. & de la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico*. Data in Brief, 25, 104344.
Disponible librement : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6710633/


**Licence :** Creative Commons Attribution 4.0 International (CC BY 4.0) — utilisation libre.

---

### 1.2 En quoi consiste-t-il ?

Le dataset contient les informations de **2111 individus** provenant de **trois pays d'Amérique latine** : le Mexique, le Pérou et la Colombie. Pour chaque individu, on dispose de **17 attributs** décrivant ses caractéristiques personnelles, ses habitudes alimentaires et son mode de vie quotidien.

L'objectif est de prédire le **niveau d'obésité** d'une personne à partir de ces informations, sans avoir besoin d'examen médical complexe ni de prise de sang. Le dataset est étiqueté selon **7 classes** correspondant aux catégories définies par l'Organisation Mondiale de la Santé (OMS) :

1. Insuffisance pondérale (Insufficient Weight)
2. Poids normal (Normal Weight)
3. Surpoids niveau I (Overweight Level I)
4. Surpoids niveau II (Overweight Level II)
5. Obésité Type I (Obesity Type I)
6. Obésité Type II (Obesity Type II)
7. Obésité Type III (Obesity Type III)

### 1.3 Description détaillée des 17 features

| Catégorie | Variable | Description | Type |
|---|---|---|---|
| **Démographie** | Gender | Sexe (Homme/Femme) | Binaire |
| | Age | Âge (14-61 ans) | Continu |
| | Height | Taille (en mètres) | Continu |
| | Weight | Poids (en kg) | Continu |
| | family_history_with_overweight | Antécédents familiaux de surpoids | Binaire |
| **Habitudes alimentaires** | FAVC | Consommation fréquente d'aliments hypercaloriques | Binaire |
| | FCVC | Fréquence de consommation de légumes | Ordinal (1-3) |
| | NCP | Nombre de repas principaux par jour | Continu (1-4) |
| | CAEC | Grignotage entre les repas | Catégoriel (4 niveaux) |
| | CH2O | Consommation quotidienne d'eau (en litres) | Continu (1-3) |
| | CALC | Consommation d'alcool | Catégoriel (4 niveaux) |
| **Mode de vie** | SMOKE | Fume | Binaire |
| | SCC | Surveille les calories consommées | Binaire |
| | FAF | Fréquence d'activité physique (h/semaine) | Continu (0-3) |
| | TUE | Temps passé sur les écrans (h/jour) | Continu (0-2) |
| | MTRANS | Moyen de transport principal | Catégoriel (5 modes) |
| **Cible** | NObesity | Niveau d'obésité (étiquette à prédire) | Catégoriel (7 classes) |

### 1.4 Origine et particularité importante des données

Les auteurs ont collecté les données via un **questionnaire en ligne** auprès de volontaires. Cependant, ils n'ont obtenu que **485 vraies réponses (23 % du dataset)**. Pour étoffer le dataset à 2111 individus, ils ont utilisé l'algorithme **SMOTE** (*Synthetic Minority Oversampling Technique*) via l'outil Weka, qui **génère artificiellement** 1626 individus synthétiques (77 % du dataset) en interpolant entre les vrais individus existants.

**Cette particularité — peu mise en avant dans la littérature existante — sera un point central de notre apport personnel.**

---

## 2. Le problème traité

### 2.1 Formulation mathématique du problème

Soit :
- $x_i \in \mathbb{R}^d$ le vecteur de features du $i$-ème individu (après encodage des variables catégorielles), avec $d \approx 23$ après one-hot encoding.
- $y_i \in \{1, 2, ..., 7\}$ la classe d'obésité de l'individu $i$.

Le problème de classification consiste à apprendre une fonction $f : \mathbb{R}^d \to \{1, ..., 7\}$ qui minimise le risque empirique :

$$\min_{\theta} \quad \frac{1}{n}\sum_{i=1}^{n} \ell(f_\theta(x_i), y_i) + \lambda \, R(\theta)$$

où $\ell$ est une fonction de perte (cross-entropy pour la régression logistique, hinge loss pour le SVM) et $R(\theta)$ un terme de régularisation (norme L1 ou L2).

### 2.2 Pourquoi ce problème est important

L'obésité est un **enjeu de santé publique majeur à l'échelle mondiale** :

- **5e cause de mortalité dans le monde** selon l'OMS
- Plus de **1 milliard de personnes obèses** dans le monde en 2024
- L'obésité est associée à des comorbidités graves : diabète de type 2, maladies cardiovasculaires, certains cancers, AVC
- Coût estimé à **$1900 milliards par an** au niveau mondial (impact économique direct et indirect)
- Le surpoids touche désormais aussi les enfants et adolescents (un enjeu majeur de prévention)

### 2.3 L'enjeu spécifique de la prédiction par mode de vie

Les méthodes traditionnelles de diagnostic reposent sur des **mesures cliniques** (IMC mesuré, analyses sanguines, bilans médicaux). L'intérêt du dataset est de proposer une **prédiction basée sur des questions simples du quotidien** : alimentation, activité physique, transport, temps d'écran...

Cela permet :
- Un **dépistage à grande échelle** sans matériel médical
- Une **prévention proactive** (identifier les profils à risque avant l'apparition de l'obésité)
- Le développement d'**applications mobiles de santé** accessibles à tous
- Une **éducation à la santé** ciblée sur les facteurs réellement prédictifs

---

## 3. Pourquoi nous nous y intéressons

### 3.1 Pertinence pour notre génération

C'est un sujet qui nous **concerne directement en tant qu'étudiants** : notre génération est exposée à toutes les variables du dataset (temps d'écran élevé, alimentation rapide, sédentarité, alcool, transport motorisé...). Nous pouvons mobiliser une **intuition personnelle** pour analyser les résultats.

### 3.2 Compréhensibilité pour un public non-médical

Contrairement à des datasets médicaux complexes (bilirubine, créatinine, marqueurs tumoraux...), **toutes les variables ici sont compréhensibles immédiatement** par n'importe qui. Cela nous permet de :
- Présenter clairement notre travail à la défense (le jury comprendra tout sans explications médicales)
- Justifier nos choix de feature engineering avec du bon sens
- Avoir un discours pédagogique solide

### 3.3 Richesse mathématique pour le cours

Le problème permet d'aborder **explicitement** tous les concepts du cours :

| Chapitre du cours | Application sur le dataset |
|---|---|
| **Ch.1 — Notions de convexité** | Les fonctions de perte (cross-entropy, hinge) sont convexes. La preuve passe par les conditions d'ordre 2 sur le Hessien. |
| **Ch.2 — Optimisation sans contrainte** | La régression logistique régularisée est un problème non contraint, strictement convexe avec L2. Conditions d'optimalité d'ordre 1 : $\nabla f(\theta^*) = 0$. |
| **Ch.3 — Optimisation sous contrainte (KKT)** | Le SVM dual est un problème quadratique convexe avec contraintes d'inégalité ($\alpha_i \geq 0$) et d'égalité ($\sum \alpha_i y_i = 0$). Application directe du théorème KKT. |

### 3.4 Existence de benchmarks

De nombreux travaux existent sur ce dataset (gradient boosting à 98 %, random forest à 96 %...). Cela nous donne :
- Des **points de comparaison** pour évaluer nos modèles convexes
- Une littérature dans laquelle nous pouvons positionner notre originalité
- La possibilité de montrer que des **modèles convexes simples** peuvent rivaliser avec des modèles complexes — un message fort

---

## 4. Notre valeur ajoutée — le cœur du projet

C'est la partie la plus importante : **ce que nous apportons en plus** par rapport aux travaux existants. Nous proposons **quatre apports concrets et originaux**.

### 4.1 Apport n°1 — Audit critique du dataset (le caractère synthétique)

#### Ce que personne ne fait
La quasi-totalité des publications utilisant ce dataset (>50 notebooks Kaggle, plusieurs articles) annonce fièrement des accuracies de **95-98 %** sans mentionner que **77 % des données sont synthétiques** (générées par SMOTE).

#### Notre démarche
1. **Séparation explicite** : nous séparons le dataset en deux parties — les 485 instances réelles vs. les 1626 instances synthétiques.
2. **Entraînement comparatif** : nous entraînons les mêmes modèles (régression logistique, SVM) sur :
   - Le dataset complet (2111 individus)
   - Les données réelles uniquement (485 individus)
   - Un sous-ensemble synthétique de même taille (485 instances générées) pour isoler l'effet de la taille
3. **Mesure de l'écart de performance** : de combien chute la précision quand on retire les données artificielles ?

#### Lien avec le cours
SMOTE génère de nouveaux points par **interpolation linéaire entre deux points existants** :
$$x_{\text{nouveau}} = \lambda x_i + (1-\lambda) x_j, \quad \lambda \in [0,1]$$

Ce qui correspond **exactement à la définition d'une combinaison convexe** (Chapitre 1, Définition 1). On peut donc montrer mathématiquement :
- Les points synthétiques se situent dans l'**enveloppe convexe** des points réels.
- Si les classes réelles ne sont pas convexes (ce qui est typique en haute dimension), SMOTE **déforme** artificiellement la frontière de décision.
- Conséquence : la complexité apparente du problème est **réduite** par SMOTE, ce qui explique les performances anormalement élevées dans la littérature.

#### Pourquoi c'est fort
C'est une critique **honnête et scientifiquement rigoureuse**. Aucun groupe d'étudiants ne pensera à faire ça, et cela montre une vraie maturité d'analyse.

---

### 4.2 Apport n°2 — Reformulation : de la classification à la régression

#### Le constat
Les 7 classes du dataset (Insuffisance, Normal, Surpoids I, II, Obésité I, II, III) ne sont **pas naturelles** : elles sont obtenues à partir des **seuils OMS appliqués à l'IMC** (Indice de Masse Corporelle = poids/taille²).

#### Notre démarche
Nous **abandonnons la classification artificielle** et nous prédisons directement l'IMC, qui est une **valeur continue**. Cela transforme le problème en une **régression**.

#### Pourquoi c'est intéressant mathématiquement

| Classification multi-classes | Régression continue |
|---|---|
| Fonction de coût discrète (accuracy) | Fonction de coût lisse (MSE), **strictement convexe** |
| Hessien difficile à analyser | $\nabla^2 f = 2 X^\top X$ (semi-défini positif) |
| Solution numérique uniquement | **Solution analytique fermée** : $\theta^* = (X^\top X)^{-1} X^\top y$ |
| Pas d'unicité garantie | **Unicité garantie** si $X^\top X$ inversible |

Cela nous permet de :
- **Démontrer explicitement** la stricte convexité de la fonction de coût
- **Calculer à la main** la solution optimale (rare et apprécié des évaluateurs)
- Vérifier les conditions d'optimalité d'ordre 1 ($\nabla f(\theta^*) = 0$) et d'ordre 2 ($\nabla^2 f \succeq 0$) **explicitement**
- Comparer Ridge (L2, strictement convexe) vs Lasso (L1, convexe non-différentiable) avec **démonstrations rigoureuses**

#### Bonus
On peut **comparer** les deux approches : la régression sur l'IMC est-elle plus stable, plus interprétable, plus précise que la classification multi-classes ? C'est une question de recherche **non triviale**.

---

### 4.3 Apport n°3 — Feature engineering personnalisé

#### Le constat
Les 17 features brutes sont parfois redondantes ou peu informatives prises individuellement. Or, **moins de features bien choisies** donne typiquement un problème **mieux conditionné** (rapport des valeurs propres du Hessien plus favorable, convergence plus rapide des algorithmes de descente de gradient — voir Ch.2).

#### Notre démarche
Nous construisons des **features composites** ayant un sens médical et comportemental :

| Feature créée | Formule | Justification |
|---|---|---|
| **IMC calculé** | poids / taille² | Vérification de cohérence avec les étiquettes officielles OMS |
| **Score sédentarité** | TUE − FAF | Capture le déséquilibre entre temps écran et activité physique |
| **Score nutrition** | FCVC − FAVC − CALC | Synthèse globale de la qualité alimentaire |
| **Régularité alimentaire** | NCP × CH2O | Combine fréquence des repas et hydratation |
| **Risque familial pondéré** | Antécédents × Âge | Combine prédisposition génétique et durée d'exposition |
| **Score lifestyle global** | combinaison pondérée des précédents | Indicateur synthétique du mode de vie |

#### Lien avec le cours
- **Réduction de dimension** → meilleur conditionnement du problème
- **Étude de la convergence** : les algorithmes du chapitre 2 convergent à vitesse $O(\kappa \log(1/\varepsilon))$ où $\kappa$ est le **conditionnement** du Hessien. En réduisant intelligemment les features, on **améliore $\kappa$** et donc la vitesse de convergence.
- **Sélection manuelle vs Lasso automatique** : on compare notre sélection (basée sur du sens) à celle obtenue automatiquement par régularisation L1.

#### Bonus
Cela nous permet de produire des **visualisations claires** (en 2D ou 3D après PCA sur ces features composites) pour la défense orale.

---

### 4.4 Apport n°4 — Étude comparative culturelle

#### Le constat
Le dataset contient des données de **3 pays différents** (Mexique, Pérou, Colombie) avec des cultures alimentaires distinctes. Pourtant, la quasi-totalité des études publiées **traite le dataset comme un tout homogène**.

#### Notre démarche
Nous segmentons le dataset par pays et nous étudions :
- Les **différences de distribution** des features entre pays
- Si un modèle entraîné sur un pays **généralise** sur un autre (problème de transfert)
- Quelles variables sont **les plus prédictives** dans chaque contexte culturel

#### Lien avec le cours
- Étude des **modèles séparés** vs **modèle global** : compromis biais-variance
- Discussion sur la **convexité de la fonction de coût agrégée** vs locale
- Possibilité d'introduire un **modèle hiérarchique** avec contrainte d'égalité (lien KKT)

---

## 5. Récapitulatif de notre démarche

### 5.1 Plan global du projet

1. **Analyse exploratoire** du dataset complet
2. **Audit critique** : séparation données réelles / synthétiques (Apport 1)
3. **Feature engineering** : construction des features composites (Apport 3)
4. **Reformulation** : classification multi-classes → régression sur IMC (Apport 2)
5. **Modélisation convexe** :
   - Régression logistique régularisée (L1, L2, ElasticNet) — démonstrations de convexité
   - SVM linéaire avec démonstration KKT explicite
   - Régression linéaire sur IMC — solution analytique
6. **Étude comparative culturelle** par pays (Apport 4)
7. **Discussion** : pourquoi les modèles convexes simples peuvent rivaliser avec les modèles complexes

### 5.2 Tableau récapitulatif des apports

| Apport | Originalité | Lien direct avec le cours |
|---|---|---|
| **1. Audit synthétique** | Critique honnête peu présente dans la littérature | Définition de combinaison convexe (Ch.1) |
| **2. Reformulation IMC** | Aucun papier ne le fait sur ce dataset | Conditions d'optimalité d'ordre 1 et 2 (Ch.2) |
| **3. Feature engineering** | Démarche basée sur du sens médical et le conditionnement | Convergence des algos (Ch.2), Lasso (Ch.1) |
| **4. Étude par pays** | Originale et culturellement pertinente | Modèles contraints, KKT (Ch.3) |

### 5.3 Pitch en une phrase pour la défense

> **"Nous prenons un dataset populaire mais utilisé naïvement par la communauté, et nous l'attaquons sous quatre angles que personne ne combine : un audit critique de la qualité réelle des données, une reformulation mathématiquement plus rigoureuse du problème, un feature engineering basé sur du sens médical, et une étude comparative entre les trois pays sources."**

---

## 6. Références principales

1. Mendoza Palechor, F., & de la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico*. Data in Brief, 25, 104344. [Lien](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6710633/)

2. De-La-Hoz-Correa, E., et al. (2019). *Obesity Level Estimation Software based on Decision Trees*. Journal of Computer Science.

3. World Health Organization. *Obesity and overweight — Key facts*. https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight

4. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. (Référence principale du cours)

5. Rockafellar, R. T. (1997). *Convex Analysis*. Princeton University Press.

6. Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321-357. (Référence pour notre Apport 1)
