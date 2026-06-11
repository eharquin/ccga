
# Reviewer 2 (no reviewer 1)
## Moderate things
+ antiwedge/antiouter is non-standard => define it somehwere => ajouté "meet" à la notation
+ Th. 2 mention points as 6-vectors, but points are only introduced as (x,y,w) at this point
+ Page 9 (below 14): we use a basis (e1, ...) but we do not precise what algebra it is. The reviewer expect it to be QC2GA, GAC or G6. What I mean to do with these notation is to stay general and therefore I do not want to precise an algebra. Find a way to make that understood.
+ page 15 "so that E^Ec = I" : the reviewer does not understand that the right complement is uniquely defined for the basis blade of the algebra. They are confused by the examples "2c = 2e123" as 2 ^ 2e123 = 4e123 (as expected). Make it clearer.
+ fig 7 states that it is the output of our code, but we do not specify the input. On devrait dire que l'input sont les intersections.

## Little things
### Pas traîté totalement / pas du tout
+ trop de pronoms là où on devrait utiliser des noms
+ pg 11, I found (25) to be a bit hidden.  Ie, this is used later, but I had SAY MORE <= je ne comprend rien à cette phrase
+ Pg 22, end of Proposition 2,  DELETE "actually" and "in fact".  And it's not clear that this paragraph after (93) is part of the proposition or not.  Ie, I suspect it's discussion and not part of the proposition, and it should be moved out of the proposition. => mon frère en christ la mise en page rend ça évident

### Traîté
+ Abstract: Delete from "Conic sections find..." to "was previously unavailable" and start with what you did ("This paper presents...")
+ enlever tous les "very"
+ Pg 4, Sect 3.1: a homogeneous polynomial of degree 2o f 3 variables" CHANGE "3" TO "three"
+ utiliser des mots pour les petits nombres => fait (sauf pour les grades et dimensions et quand on parle de 0 en tant qu'objet)
+ Abstract: "The are often encountered in domains..." change "They" to "GAs"
+ Abstract" "permit the expression of a lot of objects and operations" CHANGE TO "permit the expression of objects and operations"
+ Pg 7: "they can't" CHANGE TO "they cannot"
+ pg 7, figure 2 caption: CHANGE "3" to "Three"
+ pg 8, Theorem 2: CHANGE "by" to "through"
+ Pg 9, first paragraph: "converging methods are preferred in practice" CHANGE TO "thus converging methods are preferred in practice"
+ Pg 9, first paragraph" "These two problems are equivalent".  I assume the two problems (and I shouldn't have to assume) are finding the roots of any quartic polynomial of one variable and finding the intersection of two conics by using the degenerate conic method.  If so, I'm not sure what the point of the comment is: they just stated that Faucette did this.  If it was unclear in Faucette [11] that these problems were equivalent, then a bit more should be said here, or the phrase "These two problems are equivalent" should be deleted. => j'ai supprimé
+ Pg 9, Second paragraph: "The downside of this method" CHANGE "this" to a noun (ie, I assume they want "Richter-Bebert's method", but I'm not 100% sure; thus the need to use a noun instead of the pronoun).
+ Pg 9, "Where" after (12) should be "where"
+ Pg 9, below (13) DELETE "as displayed"
+ Pg 10, second sentence of 4.2: "It is built from" CHANGE TO "QC2A is built from"
+ pg 11, after (24), "With" should be "with"
+ pg 13, "It's important" CHANGE TO "It is important"
+ Pg 13, Def 6: "is said imaginary" CHANGE TO "is said to be imaginary"
+ Pg 14, example 1: "Consider p,p1,p2,p3 some point." CHANGE TO "Consider points p,p1,p2,p3."
+ pg 15, DELETE "A few general tools are to be defined in order to continue".
+   "For instance, this applied to a conic" CHANGE TO "Applying this norm to a conic"
+   "This is because of" CHANGE "This" to "This term is missing because of"
+ pg 15, paragraph after (63), three "this"s that should become nouns or phrases" => oui bon faut pas non plus déconner, les this sont des mots utilisables dans la langue anglaise à tout niveau de langage (j'en ai retiré 1 quand même)
+   "This has been a motivation" CHANGE TO (I think) "These problems was motivation"
+ Pg 16, Def 12: It might be easier if B_C was ordered in the order you write a conic; ie, e_infty,e1,e2,ep1,eo2,e03 => bah justement c'est ce que je fais, l'ordre c'est o1 o2 o3 1 2 i
+ pg 17, (68) The V notation was (possibly) defined in (25); possibly a word or two about V, or a backward reference to an earlier formula would be appropriate.  And I see that such a reference is made on page 18, in Theorem 9, so possibly moving that mention forward would be appropriate.
+ pg 17, paragraph before Theorem 6 is informal, especially "this would be missing the whole point".  Instead, say what the "point" is.  Yes, the "point" follows in the Theorem, but even "but the following theorem show I^{tri}_o is more than that" would be an improvement. => oui c'est vrai
+ pg 18, Theorem 9.  CHANGE "Let P a pencil of order k and C a conic not in P." TO "Let P be a pencil of order k and C be a conic not in P."
+ Pg 18, Theorem 12, proof.  CHANGE "Let P a pencil containing" TO "Let P be a pencil containing"
+ Pg 18, between Th 13 and 14.  It says "Th 11 and 13 are illustrated by Fig 5".  But the Fig 5 caption says "These figures respectively illustrate Th 9, 11 and 13."  Why no mention of Th 9 in the page 18 comment? => j'ai juste oublié
+ Pg 18, Theorem 14.  CHANGE "which means that it is the pencil P" to "which means that P V pc is the pencil P" => ça mange pas de pain ici
+ pg 20, first line, CHANGE "allows to factor any" TO "allows factoring any"
+ Pg 20, after (81), change the comma before "it is not" to a semi-colon => whaaaaaaaa
+ Pg 21, first two lines, CHANGE "6" to "six" and "3" to "three"
+ Pg 22, Section 8 CHANGE "4" to "four"
+ Pg 23, after (106), change "With" TO "with"
+ Pg 24: Algorithm 2: CHANGE "pair of lines" to "Pair of lines"
+ pg 25, Section 8.1.  CHANGE "3" to "three"
+ Pg 25, Section 9: "Fig 7 gives a few outputs of our code" CHANGE TO "Fig 7 shows a few examples computed with our code."
+ pg 27, before Section 10: "It makes it hard to tell how likely these errors are, but this also means that the algorithm is very stable and solid." CHANGE TO
 "A lack of such errors makes it hard to tell how likely these errors are, but this also suggests that the algorithm is very stable and robust." [This is the one "very" that the authors could consider leaving in their paper] => trop bien un very legit
+ Pg 27, Section 10.  Delete the opening clause of the first sentence, and start with "This paper shows the equivalence...".
+ CHANGE "a whole new set" TO "a new set"
+ CHANGE "based on pencil of conics" TO "based on pencils of conics"

# Reviewer 3
## pas changé
+ "My second general comment is about the definitions of objects such as Definition 12. Not every 1-vector is a conic and thus the definition is insufficient"
 => il est pas content de la définition 12, qui dit que les coniques duales sont les vecteurs construits sur la base (o1, o2, o3, 1, 2, i)
 => il dit que tous les 1-vecteurs ne sont pas des coniques (j'imagine qu'il sous-entend duales)
 => c'est vrai, dans l'absolu, car les 1-vecteurs qui sont des coniques duales sont ceux construits sur (o1, o2, o3, 1, 2, i)
 => il dit que la définition est donc pas suffisante
 => ma théorie: il a pas bien lu, la définition 12 décrit avec exactitude ce qui est une conique duale
-	Definition 2 and Theorem 2 - not every 1-vector is a conic or point, therefore the dimension of the resulting vector space (pencil or points) is less than 6. => Si. rien à changer.
-	Section 4 - maybe instead of equivalent, I would use isomorphic => quel enfer
-	Par. 4.4. the „direct correspondence" should be described as an isomorphism, namely a change of basis. Consequently, the last two sentences of this paragraph should be modified accordingly.
-	Theorem 3 - it is unclear to me how the claim about a sum of two points is proved. Also I note that the illustration of the set Points may be done by a cone (two lines in 2D) which corresponds to its nature better.
     => non c'est complètement con, son exemple marche pas du tout vu que des cones sont inclues dans le cône, on peut donc trouver des points qui additionnés donnent d'autres points.
-	Definition 9: so your norm is an analogue of Euclidean norm?
    => exactly
-	Paragraph after Remark 2 - you are mentioning a dual conic with no context (GA or classics). Then, a dual conic is defined in Definition 12.
    => c'est vrai, mais ça me saoûle de chambouler tout le plan. Possiblement retirer ce paragraphe?
-	Proposition 2: remove the sentence „The homogeneous coordinate…." To some remark. => jsp ce que ça veut dire le "To some remark" mais sinon fait, après pas sûr

## Changé
+ mentionner dans le texte peu importe ce qu'ils disent de mentioner avec dans texte: 
  "As the model space the authors consider QC2GA, ie., two dimensional version of geometric algebra for quadrics and argue that it is a better choice than GAC. And this is my first concern. Indeed, in Section 4the authors show that QC2GA and GAC are equivalent as algebras with identical bilinear form. This in fact means that a subalgebra of QCGA isomorphic to GAC is chosen which in the end means that those two differ only by a change of basis given by (35)-(38). This is fine but should be mentioned in the text."
  => oui, ajouté "Eqs.~(\ref{eq:equivalence_start}-\ref{eq:equivalence_end}) show by construction that QC2GA and GAC only differ by a change of basis." juste après l'align du changement de base.
+ replacer "point" par 1-PSE dans la proof 4 (parce qu'on parle pas de points mais de 1-vecteurs génériques)
-	Par. 3.2, Table 1 - radius of an ellipse? => retiré la mention de rayon, remplacé par "may be imaginary" (les ellipses imaginaires sont une chose connue)
-	Par 3.4. „0 to 4 intersection points" should be „0 to 4 real intersection points" => oui ça mange pas de pain.
-	Par. 3.4. second point, page 8: Some methods of extraction and decomposition of a degenerated conics were described in [6], maybe a reference would be appropriate at this place. => cité Richter-Gebert, par contre j'ai dû reformuler la phrase pour par que ça casse la mise en page
-	Definition 5, I have found a reference to points at infinity in GAC in 
Loučka, P., Vašík, P. Algorithms for Conic Fitting Through Given Proper and Improper Waypoints in Geometric Algebra for Conics. Adv. Appl. Clifford Algebras 34, 6 (2024). https://doi.org/10.1007/s00006-023-01308-5
Maybe you could compare these concepts and refer to the paper.
 => j'ai ajouté ce paragraphe juste après la def
    "Points at infinity are often met when working in projective geometry and conics~\cite{richter-gebert_perspectives_2011}. Points at infinity are common in the context of quadratic primitives and geometric algebra ~\cite{hrdina_geometric_2018,breuils_three-dimensional_2019} and are described in a recent paper as the meeting point of two parallel line~\cite{2024-01-09louckaAlgorithmsConicFitting}. Def.~\ref{def:pt_inf} presents points at infinity as characterised by an angle, which is the same thing."
-	Definition 8: what is a linear application? Is it orthogonal complement? Please explain. => ah mais oui en anglais c'est linear map
-	Definition 13: you are using the term anti-outer product and then in Section 7 antiwedge. Cant his be unified or explained?
-	Theorem 7 and consequent text: there should be „Let A be a n-PSE" and „Let…be…" everywhere.

-	Proof of Proposition 2: in (95) there is a redundant minus. => oui, corrigé
-	Proof of Theorem 7 - the visual form  of the proof is strange and the sentence „P is a pencil since dual…" makes not much sense.
    => ajouté du texte "\(P\) is a pencil since dual of blade of \(\mathcal{B}_C\), ie dual of outer product of dual conics, hence anti-outer product of conics"
  