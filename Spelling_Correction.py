import numpy as np

class Spelling_Correction():
    '''
    Corrects spelling mistakes by replacing the misspelled words by the closest
    words in the dictionary. It uses a BK-tree structure to store the words and look for the 
    most similar words. The distance between two words is the (weighted) edit distance. 
    
    Args:
        words (list of strings): Words of the vocabulary
        tol (int): Tolerance to calculate similar words with respect to the Edit Distance
        c_ins (int): Inserting cost for Edit Distance
        c_del (int): Deleting cost for Edit Distance
        c_rep (int): Replacing cost for Edit Distance
    
    Attributes:
        words (list of strings): Words of the vocabulary
        tol (int): Tolerance to calculate similar words with respect to the Edit Distance
        c_ins (int): Inserting cost for Edit Distance
        c_del (int): Deleting cost for Edit Distance
        c_rep (int): Replacing cost for Edit Distance
        tree (dict): BK-tree of the vocabulary 
        
    
    '''
    def __init__(self, words,tol, c_ins=1, c_del=1, c_rep=1):
        self.c_ins = c_ins
        self.c_del = c_del
        self.c_rep = c_rep
        self.words = words
        self.tol = tol

        it = iter(words)
        root = next(it)
        self.tree = (root, {})
        
        #Add words to tree
        for i in it:
            self._add_word(self.tree, i)
    
    def editDistance(self,str1,str2):
        '''
        Function that calculates the edit distance between two strings. 
        It uses less memory because we only store 2 rows instead of the whole matrix 

        Args:
            str1: (String) First string
            str2: (String) Second string
        
        Returns:
            Edit distance between str1 and str2

        '''
        m = len(str1)
        n = len(str2)

        # Create a array to memoize distance of previous computations (2 rows) 
        dist = np.zeros((2,m+1)) 

        # When second string is empty then we remove all characters 
        for i in range(m+1):
            dist[0,i] = i


        # Fill the matrix for every character of the second string
        for i in range(1,n+1):
            #This loop compares the char from second string with first string characters 
            for j in range(m+1):
                #if first string is empty then we have to perform add character peration to get second string   
                if j==0:
                    dist[i%2,j] = i

                #if character from both string is same then we do not perform any operation. We take the diagonal value

                elif str1[j-1] == str2[i-1]:
                    dist[i%2,j] = dist[(i-1)%2,j-1]

                #if the characters are different, we take the minimum of the three edits and addd it with the cost
                else:
                    del_char = dist[(i-1)%2,j] + self.c_del #Delete
                    ins_char = dist[i%2,j-1] + self.c_ins #Insert
                    rep_char = dist[(i-1)%2,j-1] + self.c_rep #Replace

                    dist[i%2,j] = min(del_char, ins_char, rep_char)

        return dist[n%2,m]


    def _add_word(self, parent, word):
        '''
        Add word word to tree
        
        Args:
            parent (dict): Parent node from where to start looking
            word (str): word we want to put in the BK-tree
        
        Returns:
            -
        '''
        pword, children = parent
        d = self.editDistance(word, pword)        
        
        #If there is not a word at distance d from the parent word, add it
        if d not in children.keys():
            children[d] = (word, {})
            
        #Otherwise, recursively find a word which does not have the same distance to the current word
        #We look at the word which is already at distance d from the parent
        else:
            self._add_word(children[d],word)
            
        
    def _search_descendants(self, parent, query_word):
        '''
        Finds words that are similar to query_word (within the tolerance)
        
        Args:
            parent (dict): Parent node from where to start looking for
            query_word (str): Word from which we want to find similar words
        
        Returns:
            results (list of strings): List of similar words to query_word within the tolerance
        '''
        
        node_word, children_dict = parent
        #Calculate the distance from the query word to the parent word
        dist_to_node = self.editDistance(query_word, node_word)

        results = []
        if dist_to_node <= self.tol: #If the parent word has a distance within the tolerance, we append it to results
            results.append((dist_to_node, node_word))
        
        #We inspect all the words with distance in [d(parent,word) - tol, d(parent,word)+tol]
        for i in range(int(dist_to_node-self.tol), int(dist_to_node+self.tol+1)):
            child = children_dict.get(i) # children_dict[i] can return keyerror
            if child is not None: #For each children within the accepted distances, start again
                results.extend(self._search_descendants(child, query_word))
                
        return results
            
    def find_closest_neighbours(self, query_word):
        '''
        Gives ordered list of similar words to query_word
        
        Args:
          query_word (str): Word that we want to compare with others
          
        Returns:
            Ordered list of similar words
        '''
        # sort by distance
        return sorted(self._search_descendants(self.tree, query_word))
    
    def correct_text(self, text):
        '''
        Corrects a text by replacing unknown words to their most similar word.
        
        Args:
            text (str): Text to correct
        
        Returns:
            correction (str): Corrected text
        '''
        correction = []
        for w in text.split(" "):
            if w in self.words:
                correction.append(w)
            else:
                w_similar = self.find_closest_neighbours(w)

                if len(w_similar)>0:
                    w_corrected = w_similar[0][1]
                    correction.append(w_corrected)
                else:
                    # no word found, simply append the unedited word
                    correction.append(w)
        return " ".join(correction)
                    
                    
