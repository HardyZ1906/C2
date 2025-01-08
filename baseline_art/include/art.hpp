/*
  Adaptive Radix Tree
  Viktor Leis, 2012
  leis@in.tum.de

  Modified by Huanchen Zhang, 2016

	Modified by Kepan Zhang (kzhang600@gatech.edu), 2024
	Changed source files to header-only form
 */
#pragma once

#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <stdint.h>    // integer types
#include <emmintrin.h> // x86 SSE intrinsics
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <deque>
#include <string>
#include <iostream>

using namespace std;

// Constants for the node types
static const int8_t NodeType4=0;
static const int8_t NodeType16=1;
static const int8_t NodeType48=2;
static const int8_t NodeType256=3;

//static
static const int8_t NodeTypeD=0;
static const int8_t NodeTypeDP=1;
static const int8_t NodeTypeF=2;
static const int8_t NodeTypeFP=3;
static const int8_t NodeTypeU=4;

// The maximum prefix length for compressed paths stored in the
// header, if the path is longer it is loaded from the database on
// demand
static const unsigned maxPrefixLength=9;

static const unsigned NodeDItemTHold=227;

// Shared header of all inner nodes
struct Node {
    // length of the compressed path (prefix)
    uint32_t prefixLength;
    // number of non-null children
    uint16_t count;
    // node type
    int8_t type;
    // compressed path (prefix)
    uint8_t prefix[maxPrefixLength];

    Node(int8_t type) : prefixLength(0),count(0),type(type) {}
};

// Node with up to 4 children
struct Node4 : Node {
    uint8_t key[4];
    Node* child[4];

    Node4() : Node(NodeType4) {
	memset(key,0,sizeof(key));
	memset(child,0,sizeof(child));
    }
};

// Node with up to 16 children
struct Node16 : Node {
    uint8_t key[16];
    Node* child[16];

    Node16() : Node(NodeType16) {
	memset(key,0,sizeof(key));
	memset(child,0,sizeof(child));
    }
};

static const uint8_t emptyMarker=48;

// Node with up to 48 children
struct Node48 : Node {
    uint8_t childIndex[256];
    Node* child[48];

    Node48() : Node(NodeType48) {
	memset(childIndex,emptyMarker,sizeof(childIndex));
	memset(child,0,sizeof(child));
    }
};

// Node with up to 256 children
struct Node256 : Node {
    Node* child[256];

    Node256() : Node(NodeType256) {
	memset(child,0,sizeof(child));
    }
};

typedef struct {
    Node* node;
    uint16_t cursor;
} NodeCursor;

//static-----------------------------------------------------
struct NodeStatic {
    int8_t type;

    NodeStatic() : type(NodeTypeFP) {}
    NodeStatic(int8_t t) : type(t) {}
};

struct NodeD : NodeStatic {
    uint8_t count;
    uint8_t data[0];

    NodeD() : NodeStatic(NodeTypeD) {}
    NodeD(uint8_t size) : NodeStatic(NodeTypeD) { count = size; }

    uint8_t* key() { return data; }
    uint8_t* key(unsigned pos) { return &data[pos]; }
    NodeStatic** child() { return (NodeStatic**)(&data[count]); }
    NodeStatic** child(unsigned pos) { return (NodeStatic**)(&data[count + sizeof(NodeStatic*) * pos]); }
};

struct NodeDP : NodeStatic {
    uint8_t count;
    uint32_t prefixLength;
    uint8_t data[0];

    NodeDP() : NodeStatic(NodeTypeDP) {}
    NodeDP(uint8_t size) : NodeStatic(NodeTypeDP) { count = size; }
    NodeDP(uint8_t size, uint32_t pl) : NodeStatic(NodeTypeDP) {
	count = size;
	prefixLength = pl;
    }

    uint8_t* prefix() { return data; }
    uint8_t* key() { return &data[prefixLength]; }
    uint8_t* key(unsigned pos) { return &data[prefixLength + pos]; }
    NodeStatic** child() {
	return (NodeStatic**)(&data[prefixLength + count]);
    }
    NodeStatic** child(unsigned pos) {
	return (NodeStatic**)(&data[prefixLength + count + sizeof(NodeStatic*) * pos]);
    }
};

struct NodeF : NodeStatic {
    uint16_t count;
    NodeStatic* child[256];

    NodeF() : NodeStatic(NodeTypeF) {}
    NodeF(uint16_t size) : NodeStatic(NodeTypeF) {
	count = size;
	memset(child, 0, sizeof(child));
    }
};

struct NodeFP : NodeStatic {
    uint16_t count;
    uint32_t prefixLength;
    uint8_t data[0];

    NodeFP() : NodeStatic(NodeTypeFP) {}
    NodeFP(uint16_t size) : NodeStatic(NodeTypeFP) { count = size; }
    NodeFP(uint16_t size, uint32_t pl) : NodeStatic(NodeTypeFP) {
	count = size;
	prefixLength = pl;
	memset(data, 0, prefixLength * sizeof(uint8_t) + 256 * sizeof(NodeStatic*));
    }

    uint8_t* prefix() { return data; }
    NodeStatic** child() { return (NodeStatic**)&data[prefixLength]; }
};

struct NodeU : NodeStatic {
    uint16_t count;
    uint32_t prefixLength;
    uint8_t prefix[maxPrefixLength];
    uint8_t data[0];

    NodeU() : NodeStatic(NodeTypeU) {}
    NodeU(uint16_t size) : NodeStatic(NodeTypeU) { count = size; }

    uint8_t* key() { return data; }
    NodeStatic** child() { return (NodeStatic**)(&data[count]); }
};

//static
typedef struct {
    NodeStatic* node;
    uint16_t cursor;
} NodeStaticCursor;

//-----------------------------------------------------------


class ART;
class CART;
class ARTIter;
class CARTIter;


class ART {

public:
    inline Node* makeLeaf(uintptr_t tid);
    inline uintptr_t getLeafValue(Node* node);
    inline bool isLeaf(Node* node);
    inline uint8_t flipSign(uint8_t keyByte);
    inline void loadKey(uintptr_t tid,uint8_t key[]);
    inline unsigned ctz(uint16_t x);

    //****************************************************************

    inline Node** findChild(Node* n,uint8_t keyByte);
    inline Node* minimum(Node* node);
    inline bool leafMatches(Node* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
    inline unsigned prefixMismatch(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength);

    inline Node* lookup(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
    inline Node* lookupPessimistic(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);

    //****************************************************************

    inline int CompareToPrefix(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength);
    inline Node* findChild_recordPath(Node* n, uint8_t keyByte, ARTIter* iter);
    inline Node* lower_bound(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength, ARTIter *iter);

    //****************************************************************

    inline unsigned min(unsigned a,unsigned b);
    inline void copyPrefix(Node* src,Node* dst);

    inline void insert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength);

    inline void insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child);
    inline void insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child);
    inline void insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child);
    inline void insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child);

    //****************************************************************

    inline void erase(Node* node,Node** nodeRef,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);

    inline void eraseNode4(Node4* node,Node** nodeRef,Node** leafPlace);
    inline void eraseNode16(Node16* node,Node** nodeRef,Node** leafPlace);
    inline void eraseNode48(Node48* node,Node** nodeRef,uint8_t keyByte);
    inline void eraseNode256(Node256* node,Node** nodeRef,uint8_t keyByte);


    ART();
    ART(unsigned kl);
    ART(Node* r);
    ART(Node* r, unsigned kl);

    void load(vector<string> &keys, vector<uint64_t> &values, unsigned maxKeyLength);
    void load(vector<uint64_t> &keys, vector<uint64_t> &values);
    void insert(uint8_t key[], uintptr_t value, unsigned maxKeyLength);
    uint64_t lookup(uint8_t key[], unsigned keyLength, unsigned maxKeyLength);
    uint64_t lookup(uint64_t key64);
    bool lower_bound(uint8_t key[], unsigned keyLength, unsigned maxKeyLength, ARTIter* iter);
    bool lower_bound(uint64_t key64, ARTIter* iter);
    void erase(uint8_t key[], unsigned keyLength, unsigned maxKeyLength);
    uint64_t getMemory();

    friend class ARTIter;

private:
    Node* root;
    unsigned key_length;

    //stats
    uint64_t memory;
    uint64_t num_items;
    uint64_t node4_count;
    uint64_t node16_count;
    uint64_t node48_count;
    uint64_t node256_count;

    // This address is used to communicate that search failed
    Node* nullNode;
};

class CART {

public:
    inline Node* makeLeaf(uintptr_t tid);
    inline uintptr_t getLeafValue(Node* node);
    inline uintptr_t getLeafValue(NodeStatic* node);
    inline bool isLeaf(Node* node);
    inline bool isLeaf(NodeStatic* node);
    inline uint8_t flipSign(uint8_t keyByte);
    inline void loadKey(uintptr_t tid,uint8_t key[]);
    inline unsigned ctz(uint16_t x);

    //****************************************************************

    inline Node** findChild(Node* n,uint8_t keyByte);
    inline NodeStatic** findChild(NodeStatic* n,uint8_t keyByte);
    inline Node* minimum(Node* node);
    inline NodeStatic* minimum(NodeStatic* node);
    inline bool leafMatches(Node* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
    inline bool leafMatches(NodeStatic* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
    inline unsigned prefixMismatch(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength);
    inline unsigned prefixMismatch(NodeStatic* node,uint8_t key[],unsigned depth,unsigned maxKeyLength);

    inline NodeStatic* lookup(NodeStatic* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);

    //****************************************************************

    inline int CompareToPrefix(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength);
    inline int CompareToPrefix(NodeStatic* n,uint8_t key[],unsigned depth,unsigned maxKeyLength);
    inline NodeStatic* findChild_recordPath(NodeStatic* n, uint8_t keyByte, CARTIter* iter);
    //inline Node* findChild_recordPath(Node* n, uint8_t keyByte, ARTIter* iter);
    inline NodeStatic* lower_bound(NodeStatic* node, uint8_t key[], unsigned keyLength, unsigned depth, unsigned maxKeyLength, CARTIter* iter);
    //inline Node* lower_bound(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength, ARTIter *iter);

    //****************************************************************

    inline unsigned min(unsigned a,unsigned b);
    inline void copyPrefix(Node* src,Node* dst);

    inline void insert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength);

    inline void insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child);
    inline void insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child);
    inline void insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child);
    inline void insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child);

    //****************************************************************

    inline void Node4_to_NodeD(Node4* n, NodeD* n_static);
    inline void Node16_to_NodeD(Node16* n, NodeD* n_static);
    inline void Node48_to_NodeD(Node48* n, NodeD* n_static);
    inline void Node256_to_NodeD(Node256* n, NodeD* n_static);
    inline void Node_to_NodeD(Node* n, NodeD* n_static);

    inline void Node4_to_NodeDP(Node4* n, NodeDP* n_static);
    inline void Node16_to_NodeDP(Node16* n, NodeDP* n_static);
    inline void Node48_to_NodeDP(Node48* n, NodeDP* n_static);
    inline void Node256_to_NodeDP(Node256* n, NodeDP* n_static);
    inline void Node_to_NodeDP(Node* n, NodeDP* n_static);

    inline void Node4_to_NodeF(Node4* n, NodeF* n_static);
    inline void Node16_to_NodeF(Node16* n, NodeF* n_static);
    inline void Node48_to_NodeF(Node48* n, NodeF* n_static);
    inline void Node256_to_NodeF(Node256* n, NodeF* n_static);
    inline void Node_to_NodeF(Node* n, NodeF* n_static);

    inline void Node4_to_NodeFP(Node4* n, NodeFP* n_static);
    inline void Node16_to_NodeFP(Node16* n, NodeFP* n_static);
    inline void Node48_to_NodeFP(Node48* n, NodeFP* n_static);
    inline void Node256_to_NodeFP(Node256* n, NodeFP* n_static);
    inline void Node_to_NodeFP(Node* n, NodeFP* n_static);

    inline size_t node_size(NodeStatic* n);
    inline NodeStatic* convert_to_static();
    inline NodeStatic* convert_to_static(Node* n);


    CART();
    CART(unsigned kl);
    CART(Node* r);
    CART(Node* r, unsigned kl);

    void load(vector<string> &keys, vector<uint64_t> &values, unsigned maxKeyLength);
    void load(vector<uint64_t> &keys, vector<uint64_t> &values);
    void insert(uint8_t key[], uintptr_t value, unsigned maxKeyLength);
    void convert();
    uint64_t lookup(uint8_t key[], unsigned keyLength, unsigned maxKeyLength);
    uint64_t lookup(uint64_t key64);
    bool lower_bound(uint8_t key[], unsigned keyLength, unsigned maxKeyLength, CARTIter* iter);
    bool lower_bound(uint64_t key64, CARTIter* iter);
    uint64_t getMemory();

    friend class CARTIter;

private:
    Node* root;
    NodeStatic* static_root;
    unsigned key_length;

    //stats
    uint64_t memory;
    uint64_t static_memory;
    uint64_t num_items;
    uint64_t node4_count;
    uint64_t node16_count;
    uint64_t node48_count;
    uint64_t node256_count;
    uint64_t nodeD_count;
    uint64_t nodeDP_count;
    uint64_t nodeF_count;
    uint64_t nodeFP_count;

    // This address is used to communicate that search failed
    Node* nullNode;
    NodeStatic* nullNode_static;
};

class ARTIter {
public:
    inline Node* minimum_recordPath(Node* node);
    inline Node* nextSlot();
    inline Node* currentLeaf();
    inline Node* nextLeaf();

    ARTIter();
    ARTIter(ART* idx);

    uint64_t value();
    bool operator ++ (int);

    friend class ART;

private:
    ART* index;
    std::vector<NodeCursor> node_stack;
    uint64_t val;
};

class CARTIter {
public:
    inline NodeStatic* minimum_recordPath(NodeStatic* node);
    inline NodeStatic* nextSlot();
    inline NodeStatic* currentLeaf();
    inline NodeStatic* nextLeaf();

    CARTIter();
    CARTIter(CART* idx);

    uint64_t value();
    bool operator ++ (int);

    friend class CART;

private:
    CART* index;
    std::vector<NodeStaticCursor> node_stack;
    uint64_t val;
};

// #define LOADKEY_INT 1

//********************************************************************
// ARTIter
//********************************************************************

inline Node* ART::makeLeaf(uintptr_t tid) {
    // Create a pseudo-leaf
    return reinterpret_cast<Node*>((tid<<1)|1);
}

inline uintptr_t ART::getLeafValue(Node* node) {
    // The the value stored in the pseudo-leaf
    return reinterpret_cast<uintptr_t>(node)>>1;
}

inline bool ART::isLeaf(Node* node) {
    // Is the node a leaf?
    return reinterpret_cast<uintptr_t>(node)&1;
}

inline uint8_t ART::flipSign(uint8_t keyByte) {
    // Flip the sign bit, enables signed SSE comparison of unsigned values, used by Node16
    return keyByte^128;
}

#ifdef LOADKEY_INT
inline void ART::loadKey(uintptr_t tid,uint8_t key[]) {
    // Store the key of the tuple into the key vector
    // Implementation is database specific
    reinterpret_cast<uint64_t*>(key)[0]=__builtin_bswap64(tid);
}
#else
inline void ART::loadKey(uintptr_t tid,uint8_t key[]) {
    memcpy(reinterpret_cast<void*>(key), (const void*)tid, key_length);
}
#endif

inline unsigned ART::ctz(uint16_t x) {
    // Count trailing zeros, only defined for x>0
#ifdef __GNUC__
    return __builtin_ctz(x);
#else
    // Adapted from Hacker's Delight
    unsigned n=1;
    if ((x&0xFF)==0) {n+=8; x=x>>8;}
    if ((x&0x0F)==0) {n+=4; x=x>>4;}
    if ((x&0x03)==0) {n+=2; x=x>>2;}
    return n-(x&1);
#endif
}

inline Node** ART::findChild(Node* n,uint8_t keyByte) {
    // Find the next child for the keyByte
    switch (n->type) {
    case NodeType4: {
	Node4* node=static_cast<Node4*>(n);
	for (unsigned i=0;i<node->count;i++)
	    if (node->key[i]==keyByte)
		return &node->child[i];
	return &nullNode;
    }
    case NodeType16: {
	Node16* node=static_cast<Node16*>(n);
	__m128i cmp=_mm_cmpeq_epi8(_mm_set1_epi8(flipSign(keyByte)),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
	unsigned bitfield=_mm_movemask_epi8(cmp)&((1<<node->count)-1);
	if (bitfield)
	    return &node->child[ctz(bitfield)]; else
	    return &nullNode;
    }
    case NodeType48: {
	Node48* node=static_cast<Node48*>(n);
	if (node->childIndex[keyByte]!=emptyMarker)
	    return &node->child[node->childIndex[keyByte]]; else
	    return &nullNode;
    }
    case NodeType256: {
	Node256* node=static_cast<Node256*>(n);
	return &(node->child[keyByte]);
    }
    }
    throw; // Unreachable
}

inline Node* ART::minimum(Node* node) {
    // Find the leaf with smallest key
    if (!node)
	return NULL;

    if (isLeaf(node))
	return node;

    switch (node->type) {
    case NodeType4: {
	Node4* n=static_cast<Node4*>(node);
	return minimum(n->child[0]);
    }
    case NodeType16: {
	Node16* n=static_cast<Node16*>(node);
	return minimum(n->child[0]);
    }
    case NodeType48: {
	Node48* n=static_cast<Node48*>(node);
	unsigned pos=0;
	while (n->childIndex[pos]==emptyMarker)
	    pos++;
	return minimum(n->child[n->childIndex[pos]]);
    }
    case NodeType256: {
	Node256* n=static_cast<Node256*>(node);
	unsigned pos=0;
	while (!n->child[pos])
	    pos++;
	return minimum(n->child[pos]);
    }
    }
    throw; // Unreachable
}

inline bool ART::leafMatches(Node* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    // Check if the key of the leaf is equal to the searched key
    if (depth!=keyLength) {
	uint8_t leafKey[maxKeyLength];
	loadKey(getLeafValue(leaf),leafKey);
	for (unsigned i=depth;i<keyLength;i++)
	    if (leafKey[i]!=key[i])
		return false;
    }
    return true;
}

inline unsigned ART::prefixMismatch(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
    // Compare the key with the prefix of the node, return the number matching bytes
    unsigned pos;
    if (node->prefixLength>maxPrefixLength) {
	for (pos=0;pos<maxPrefixLength;pos++)
	    if (key[depth+pos]!=node->prefix[pos])
		return pos;
	uint8_t minKey[maxKeyLength];
	loadKey(getLeafValue(minimum(node)),minKey);
	for (;pos<node->prefixLength;pos++)
	    if (key[depth+pos]!=minKey[depth+pos])
		return pos;
    } else {
	for (pos=0;pos<node->prefixLength;pos++)
	    if (key[depth+pos]!=node->prefix[pos])
		return pos;
    }
    return pos;
}

inline Node* ART::lookup(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    // Find the node with a matching key, optimistic version

    bool skippedPrefix=false; // Did we optimistically skip some prefix without checking it?

    while (node!=NULL) {
	if (isLeaf(node)) {
	    if (!skippedPrefix&&depth==keyLength) // No check required
		return node;

	    if (depth!=keyLength) {
		// Check leaf
		uint8_t leafKey[maxKeyLength];
		loadKey(getLeafValue(node),leafKey);
		for (unsigned i=(skippedPrefix?0:depth);i<keyLength;i++)
		    if (leafKey[i]!=key[i])
			return NULL;
	    }
	    return node;
	}

	if (node->prefixLength) {
	    if (node->prefixLength<maxPrefixLength) {
		for (unsigned pos=0;pos<node->prefixLength;pos++)
		    if (key[depth+pos]!=node->prefix[pos])
			return NULL;
	    } else
		skippedPrefix=true;
	    depth+=node->prefixLength;
	}

	node=*findChild(node,key[depth]);
	depth++;
    }

    return NULL;
}

inline Node* ART::lookupPessimistic(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    // Find the node with a matching key, alternative pessimistic version

    while (node!=NULL) {
	if (isLeaf(node)) {
	    if (leafMatches(node,key,keyLength,depth,maxKeyLength))
		return node;
	    return NULL;
	}

	if (prefixMismatch(node,key,depth,maxKeyLength)!=node->prefixLength)
	    return NULL; else
	    depth+=node->prefixLength;

	node=*findChild(node,key[depth]);
	depth++;
    }

    return NULL;
}

inline unsigned ART::min(unsigned a,unsigned b) {
    // Helper function
    return (a<b)?a:b;
}

inline void ART::copyPrefix(Node* src,Node* dst) {
    // Helper function that copies the prefix from the source to the destination node
    dst->prefixLength=src->prefixLength;
    memcpy(dst->prefix,src->prefix,min(src->prefixLength,maxPrefixLength));
}

inline void ART::insert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength) {
    // Insert the leaf value into the tree

    if (node==NULL) {
	*nodeRef=makeLeaf(value);
	return;
    }

    if (isLeaf(node)) {
	// Replace leaf with Node4 and store both leaves in it
	uint8_t existingKey[maxKeyLength];
	loadKey(getLeafValue(node),existingKey);
	unsigned newPrefixLength=0;
	while (existingKey[depth+newPrefixLength]==key[depth+newPrefixLength])
	    newPrefixLength++;

	Node4* newNode=new Node4();
	memory += sizeof(Node4); //h
	node4_count++; //h
	newNode->prefixLength=newPrefixLength;
	memcpy(newNode->prefix,key+depth,min(newPrefixLength,maxPrefixLength));
	*nodeRef=newNode;

	insertNode4(newNode,nodeRef,existingKey[depth+newPrefixLength],node);
	insertNode4(newNode,nodeRef,key[depth+newPrefixLength],makeLeaf(value));
	num_items++; //h
	return;
    }

    // Handle prefix of inner node
    if (node->prefixLength) {
	unsigned mismatchPos=prefixMismatch(node,key,depth,maxKeyLength);
	if (mismatchPos!=node->prefixLength) {
	    // Prefix differs, create new node
	    Node4* newNode=new Node4();
	    memory += sizeof(Node4); //h
	    node4_count++; //h
	    *nodeRef=newNode;
	    newNode->prefixLength=mismatchPos;
	    memcpy(newNode->prefix,node->prefix,min(mismatchPos,maxPrefixLength));
	    // Break up prefix
	    if (node->prefixLength<maxPrefixLength) {
		insertNode4(newNode,nodeRef,node->prefix[mismatchPos],node);
		node->prefixLength-=(mismatchPos+1);
		memmove(node->prefix,node->prefix+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
	    } else {
		node->prefixLength-=(mismatchPos+1);
		uint8_t minKey[maxKeyLength];
		loadKey(getLeafValue(minimum(node)),minKey);
		insertNode4(newNode,nodeRef,minKey[depth+mismatchPos],node);
		memmove(node->prefix,minKey+depth+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
	    }
	    insertNode4(newNode,nodeRef,key[depth+mismatchPos],makeLeaf(value));
	    num_items++; //h
	    return;
	}
	depth+=node->prefixLength;
    }

    // Recurse
    Node** child=findChild(node,key[depth]);
    if (*child) {
	insert(*child,child,key,depth+1,value,maxKeyLength);
	return;
    }

    // Insert leaf into inner node
    Node* newNode=makeLeaf(value);
    switch (node->type) {
    case NodeType4: insertNode4(static_cast<Node4*>(node),nodeRef,key[depth],newNode); break;
    case NodeType16: insertNode16(static_cast<Node16*>(node),nodeRef,key[depth],newNode); break;
    case NodeType48: insertNode48(static_cast<Node48*>(node),nodeRef,key[depth],newNode); break;
    case NodeType256: insertNode256(static_cast<Node256*>(node),nodeRef,key[depth],newNode); break;
    }
    num_items++; //h
}

inline void ART::insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    if (node->count<4) {
	// Insert element
	unsigned pos;
	for (pos=0;(pos<node->count)&&(node->key[pos]<keyByte);pos++);
	memmove(node->key+pos+1,node->key+pos,node->count-pos);
	memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t));
	node->key[pos]=keyByte;
	node->child[pos]=child;
	node->count++;
    } else {
	// Grow to Node16
	Node16* newNode=new Node16();
	memory += sizeof(Node16); //h
	node16_count++; //h
	*nodeRef=newNode;
	newNode->count=4;
	copyPrefix(node,newNode);
	for (unsigned i=0;i<4;i++)
	    newNode->key[i]=flipSign(node->key[i]);
	memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
	delete node;
	memory -= sizeof(Node4); //h
	node4_count--; //h
	return insertNode16(newNode,nodeRef,keyByte,child);
    }
}

inline void ART::insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    if (node->count<16) {
	// Insert element
	uint8_t keyByteFlipped=flipSign(keyByte);
	__m128i cmp=_mm_cmplt_epi8(_mm_set1_epi8(keyByteFlipped),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
	uint16_t bitfield=_mm_movemask_epi8(cmp)&(0xFFFF>>(16-node->count));
	unsigned pos=bitfield?ctz(bitfield):node->count;
	memmove(node->key+pos+1,node->key+pos,node->count-pos);
	memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t));
	node->key[pos]=keyByteFlipped;
	node->child[pos]=child;
	node->count++;
    } else {
	// Grow to Node48
	Node48* newNode=new Node48();
	memory += sizeof(Node48); //h
	node48_count++; //h
	*nodeRef=newNode;
	memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
	for (unsigned i=0;i<node->count;i++)
	    newNode->childIndex[flipSign(node->key[i])]=i;
	copyPrefix(node,newNode);
	newNode->count=node->count;
	delete node;
	memory -= sizeof(Node16); //h
	node16_count--; //h
	return insertNode48(newNode,nodeRef,keyByte,child);
    }
}

inline void ART::insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    if (node->count<48) {
	// Insert element
	unsigned pos=node->count;
	if (node->child[pos])
	    for (pos=0;node->child[pos]!=NULL;pos++);
	node->child[pos]=child;
	node->childIndex[keyByte]=pos;
	node->count++;
    } else {
	// Grow to Node256
	Node256* newNode=new Node256();
	memory += sizeof(Node256); //h
	node256_count++; //h
	for (unsigned i=0;i<256;i++)
	    if (node->childIndex[i]!=48)
		newNode->child[i]=node->child[node->childIndex[i]];
	newNode->count=node->count;
	copyPrefix(node,newNode);
	*nodeRef=newNode;
	delete node;
	memory -= sizeof(Node48); //h
	node48_count--; //h
	return insertNode256(newNode,nodeRef,keyByte,child);
    }
}

inline void ART::insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    node->count++;
    node->child[keyByte]=child;
}


inline void ART::erase(Node* node,Node** nodeRef,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    // Delete a leaf from a tree

    if (!node)
	return;

    if (isLeaf(node)) {
	// Make sure we have the right leaf
	if (leafMatches(node,key,keyLength,depth,maxKeyLength))
	    *nodeRef=NULL;
	return;
    }

    // Handle prefix
    if (node->prefixLength) {
	if (prefixMismatch(node,key,depth,maxKeyLength)!=node->prefixLength)
	    return;
	depth+=node->prefixLength;
    }

    Node** child=findChild(node,key[depth]);
    if (isLeaf(*child)&&leafMatches(*child,key,keyLength,depth,maxKeyLength)) {
	// Leaf found, delete it in inner node
	switch (node->type) {
	case NodeType4: eraseNode4(static_cast<Node4*>(node),nodeRef,child); break;
	case NodeType16: eraseNode16(static_cast<Node16*>(node),nodeRef,child); break;
	case NodeType48: eraseNode48(static_cast<Node48*>(node),nodeRef,key[depth]); break;
	case NodeType256: eraseNode256(static_cast<Node256*>(node),nodeRef,key[depth]); break;
	}
    } else {
	//Recurse
	erase(*child,child,key,keyLength,depth+1,maxKeyLength);
    }
}

inline void ART::eraseNode4(Node4* node,Node** nodeRef,Node** leafPlace) {
    // Delete leaf from inner node
    unsigned pos=leafPlace-node->child;
    memmove(node->key+pos,node->key+pos+1,node->count-pos-1);
    memmove(node->child+pos,node->child+pos+1,(node->count-pos-1)*sizeof(uintptr_t));
    node->count--;

    if (node->count==1) {
	// Get rid of one-way node
	Node* child=node->child[0];
	if (!isLeaf(child)) {
	    // Concantenate prefixes
	    unsigned l1=node->prefixLength;
	    if (l1<maxPrefixLength) {
		node->prefix[l1]=node->key[0];
		l1++;
	    }
	    if (l1<maxPrefixLength) {
		unsigned l2=min(child->prefixLength,maxPrefixLength-l1);
		memcpy(node->prefix+l1,child->prefix,l2);
		l1+=l2;
	    }
	    // Store concantenated prefix
	    memcpy(child->prefix,node->prefix,min(l1,maxPrefixLength));
	    child->prefixLength+=node->prefixLength+1;
	}
	*nodeRef=child;
	delete node;
	memory -= sizeof(Node4); //h
	node4_count--; //h
    }
}

inline void ART::eraseNode16(Node16* node,Node** nodeRef,Node** leafPlace) {
    // Delete leaf from inner node
    unsigned pos=leafPlace-node->child;
    memmove(node->key+pos,node->key+pos+1,node->count-pos-1);
    memmove(node->child+pos,node->child+pos+1,(node->count-pos-1)*sizeof(uintptr_t));
    node->count--;

    if (node->count==3) {
	// Shrink to Node4
	Node4* newNode=new Node4();
	memory += sizeof(Node4); //h
	node4_count++; //h
	newNode->count=node->count;
	copyPrefix(node,newNode);
	for (unsigned i=0;i<4;i++)
	    newNode->key[i]=flipSign(node->key[i]);
	memcpy(newNode->child,node->child,sizeof(uintptr_t)*4);
	*nodeRef=newNode;
	delete node;
	memory -= sizeof(Node16); //h
	node16_count--; //h
    }
}

inline void ART::eraseNode48(Node48* node,Node** nodeRef,uint8_t keyByte) {
    // Delete leaf from inner node
    node->child[node->childIndex[keyByte]]=NULL;
    node->childIndex[keyByte]=emptyMarker;
    node->count--;

    if (node->count==12) {
	// Shrink to Node16
	Node16 *newNode=new Node16();
	memory += sizeof(Node16); //h
	node16_count++; //h
	*nodeRef=newNode;
	copyPrefix(node,newNode);
	for (unsigned b=0;b<256;b++) {
	    if (node->childIndex[b]!=emptyMarker) {
		newNode->key[newNode->count]=flipSign(b);
		newNode->child[newNode->count]=node->child[node->childIndex[b]];
		newNode->count++;
	    }
	}
	delete node;
	memory -= sizeof(Node48); //h
	node48_count--; //h
    }
}

inline void ART::eraseNode256(Node256* node,Node** nodeRef,uint8_t keyByte) {
    // Delete leaf from inner node
    node->child[keyByte]=NULL;
    node->count--;

    if (node->count==37) {
	// Shrink to Node48
	Node48 *newNode=new Node48();
	memory += sizeof(Node48); //h
	node48_count++; //h
	*nodeRef=newNode;
	copyPrefix(node,newNode);
	for (unsigned b=0;b<256;b++) {
	    if (node->child[b]) {
		newNode->childIndex[b]=newNode->count;
		newNode->child[newNode->count]=node->child[b];
		newNode->count++;
	    }
	}
	delete node;
	memory -= sizeof(Node256); //h
	node256_count--; //h
    }
}

inline int ART::CompareToPrefix(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
    unsigned pos;
    if (node->prefixLength>maxPrefixLength) {
	for (pos=0;pos<maxPrefixLength;pos++) {
	    if (key[depth+pos]!=node->prefix[pos]) {
		if (key[depth+pos]>node->prefix[pos])
		    return 1;
		else
		    return -1;
	    }
	}
	uint8_t minKey[maxKeyLength];
	loadKey(getLeafValue(minimum(node)),minKey);
	for (;pos<node->prefixLength;pos++) {
	    if (key[depth+pos]!=minKey[depth+pos]) {
		if (key[depth+pos]>minKey[depth+pos])
		    return 1;
		else
		    return -1;
	    }
	}
    } else {
	for (pos=0;pos<node->prefixLength;pos++) {
	    if (key[depth+pos]!=node->prefix[pos]) {
		if (key[depth+pos]>node->prefix[pos])
		    return 1;
		else
		    return -1;
	    }
	}
    }
    return 0;
}

inline Node* ART::findChild_recordPath(Node* n, uint8_t keyByte, ARTIter* iter) {
    NodeCursor nc;
    nc.node = n;
    switch (n->type) {
    case NodeType4: {
	Node4* node=static_cast<Node4*>(n);
	for (unsigned i=0;i<node->count;i++) {
	    if (node->key[i]>=keyByte) {
		nc.cursor = i;
		iter->node_stack.push_back(nc);
		if (node->key[i]==keyByte)
		    return node->child[i];
		else
		    return iter->minimum_recordPath(node->child[i]);
	    }
	}
	iter->node_stack.pop_back();
	//return iter->minimum_recordPath(iter->nextSlot());
	return iter->nextLeaf();
    }
    case NodeType16: {
	Node16* node=static_cast<Node16*>(n);
	for (unsigned i=0;i<node->count;i++) {
	    //if (node->key[i]>=keyByte) { THE BUGGGGGG!
	    if (node->key[i]>=flipSign(keyByte)) {
		nc.cursor = i;
		iter->node_stack.push_back(nc);
		//if (node->key[i]==keyByte) THE BUGGGGGG!
		if (node->key[i]==flipSign(keyByte))
		    return node->child[i];
		else
		    return iter->minimum_recordPath(node->child[i]);
	    }
	}
	iter->node_stack.pop_back();
	//return iter->minimum_recordPath(iter->nextSlot());
	return iter->nextLeaf();
    }
    case NodeType48: {
	Node48* node=static_cast<Node48*>(n);
	if (node->childIndex[keyByte]!=emptyMarker) {
	    nc.cursor = keyByte;
	    iter->node_stack.push_back(nc);
	    return node->child[node->childIndex[keyByte]]; 
	}
	else {
	    for (unsigned i=keyByte; i<256; i++) {
		if (node->childIndex[i]!=emptyMarker) {
		    nc.cursor = i;
		    iter->node_stack.push_back(nc);
		    return node->child[node->childIndex[i]]; 
		}	  
	    }
	    iter->node_stack.pop_back();
	    //return iter->minimum_recordPath(iter->nextSlot());
	    return iter->nextLeaf();
	}
    }
    case NodeType256: {
	Node256* node=static_cast<Node256*>(n);
	if (node->child[keyByte]!=NULL) {
	    nc.cursor = keyByte;
	    iter->node_stack.push_back(nc);
	    return node->child[keyByte];
	}
	else {
	    for (unsigned i=keyByte; i<256; i++) {
		if (node->child[i]!=NULL) {
		    nc.cursor = i;
		    iter->node_stack.push_back(nc);
		    return node->child[i]; 
		}	  
	    }
	    iter->node_stack.pop_back();
	    //return iter->minimum_recordPath(iter->nextSlot());
	    return iter->nextLeaf();
	}
    }
    }
    throw; // Unreachable
}

inline Node* ART::lower_bound(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength, ARTIter *iter) {
    iter->node_stack.clear();
    while (node!=NULL) {
	if (isLeaf(node)) {
	    return node;
	}

	int ctp = CompareToPrefix(node,key,depth,maxKeyLength);
	depth+=node->prefixLength;

	if (ctp > 0) {
	    iter->node_stack.pop_back();
	    //return iter->minimum_recordPath(iter->nextSlot());
	    return iter->nextLeaf();
	}
	else if (ctp < 0) {
	    return iter->minimum_recordPath(node);
	}

	node = findChild_recordPath(node, key[depth], iter);
	depth++;
    }

    return NULL;
}


ART::ART() 
    : root(NULL), key_length(8), memory(0), num_items(0), 
      node4_count(0), node16_count(0), node48_count(0), node256_count(0), nullNode(NULL)
{}

ART::ART(unsigned kl) 
    : root(NULL), key_length(kl), memory(0), num_items(0), 
      node4_count(0), node16_count(0), node48_count(0), node256_count(0), nullNode(NULL)
{}

ART::ART(Node* r) 
    : root(r), key_length(8), memory(0), num_items(0), 
      node4_count(0), node16_count(0), node48_count(0), node256_count(0), nullNode(NULL)
{}

ART::ART(Node* r, unsigned kl) 
    : root(r), key_length(kl), memory(0), num_items(0), 
      node4_count(0), node16_count(0), node48_count(0), node256_count(0), nullNode(NULL)
{}

void ART::load(vector<string> &keys, vector<uint64_t> &values, unsigned maxKeyLength) {
    uint8_t key[maxKeyLength];
    for (int i = 0; i < keys.size(); i++) {
	loadKey((uintptr_t)keys[i].c_str(), key);
	insert(key, values[i], key_length);
    }
}

void ART::load(vector<uint64_t> &keys, vector<uint64_t> &values) {
    uint8_t key[8];
    for (int i = 0; i < keys.size(); i++) {
	loadKey(keys[i], key);
	insert(key, values[i], key_length);
    }
}

void ART::insert(uint8_t key[], uintptr_t value, unsigned maxKeyLength) {
    insert(root, &root, key, 0, value, maxKeyLength);
}

uint64_t ART::lookup(uint8_t key[], unsigned keyLength, unsigned maxKeyLength) {
    Node* leaf = lookup(root, key, keyLength, 0, maxKeyLength);
    if (isLeaf(leaf))
	return getLeafValue(leaf);
    return (uint64_t)0;
}

uint64_t ART::lookup(uint64_t key64) {
    uint8_t key[8];
    loadKey(key64, key);
    Node* leaf = lookup(root, key, 8, 0, 8);
    if (isLeaf(leaf))
	return getLeafValue(leaf);
    return (uint64_t)0;
}

bool ART::lower_bound(uint8_t key[], unsigned keyLength, unsigned maxKeyLength, ARTIter* iter) {
    Node* leaf = lower_bound(root, key, keyLength, 0, maxKeyLength, iter);
    if (isLeaf(leaf)) {
	iter->val = (uint64_t)getLeafValue(leaf);
	iter->nextLeaf();
	return true;
    }
    else {
	iter->val = 0;
	return false;
    }
}

bool ART::lower_bound(uint64_t key64, ARTIter* iter) {
    uint8_t key[8];
    loadKey(key64, key);
    Node* leaf = lower_bound(root, key, 8, 0, 8, iter);
    if (isLeaf(leaf)) {
	iter->val = (uint64_t)getLeafValue(leaf);
	iter->nextLeaf();
	return true;
    }
    else {
	iter->val = 0;
	return false;
    }
}


void ART::erase(uint8_t key[], unsigned keyLength, unsigned maxKeyLength) {
    erase(root, &root, key, keyLength, 0, maxKeyLength);
}

uint64_t ART::getMemory() {
    return memory;
}


//********************************************************************
// ARTIter
//********************************************************************
ARTIter::ARTIter() {
    index = NULL;
    val = 0;
}

ARTIter::ARTIter(ART* idx) {
    index = idx;
    val = 0;
}


inline Node* ARTIter::minimum_recordPath(Node* node) {
    if (!node)
	return NULL;

    if (index->isLeaf(node))
	return node;

    NodeCursor nc;
    nc.node = node;
    nc.cursor = 0;
    node_stack.push_back(nc);

    switch (node->type) {
    case NodeType4: {
	Node4* n=static_cast<Node4*>(node);
	return minimum_recordPath(n->child[0]);
    }
    case NodeType16: {
	Node16* n=static_cast<Node16*>(node);
	return minimum_recordPath(n->child[0]);
    }
    case NodeType48: {
	Node48* n=static_cast<Node48*>(node);
	unsigned pos=0;
	while (n->childIndex[pos]==emptyMarker)
	    pos++;
	node_stack.back().cursor = pos;
	return minimum_recordPath(n->child[n->childIndex[pos]]);
    }
    case NodeType256: {
	Node256* n=static_cast<Node256*>(node);
	unsigned pos=0;
	while (!n->child[pos])
	    pos++;
	node_stack.back().cursor = pos;
	return minimum_recordPath(n->child[pos]);
    }
    }
    throw; // Unreachable
}

inline Node* ARTIter::nextSlot() {
    //while (node_stack.empty()) { THE BUGGGGGG!
    while (!node_stack.empty()) {
	Node* n = node_stack.back().node;
	uint16_t cursor = node_stack.back().cursor;
	cursor++;
	node_stack.back().cursor = cursor;
	switch (n->type) {
	case NodeType4: {
	    Node4* node=static_cast<Node4*>(n);
	    if (cursor < node->count)
		return node->child[cursor];
	    break;
	}
	case NodeType16: {
	    Node16* node=static_cast<Node16*>(n);
	    if (cursor < node->count)
		return node->child[cursor];
	    break;
	}
	case NodeType48: {
	    Node48* node=static_cast<Node48*>(n);
	    for (unsigned i=cursor; i<256; i++)
		if (node->childIndex[i]!=emptyMarker) {
		    node_stack.back().cursor = i;
		    return node->child[node->childIndex[i]];
		}
	    break;
	}
	case NodeType256: {
	    Node256* node=static_cast<Node256*>(n);
	    for (unsigned i=cursor; i<256; i++)
		if (node->child[i]!=NULL) {
		    node_stack.back().cursor = i;
		    return node->child[i]; 
		}
	    break;
	}
	}
	node_stack.pop_back();
    }
    return NULL;
}

inline Node* ARTIter::currentLeaf() {
    if (node_stack.size() == 0)
	return NULL;

    Node* n = node_stack.back().node;
    uint16_t cursor = node_stack.back().cursor;

    switch (n->type) {
    case NodeType4: {
	Node4* node=static_cast<Node4*>(n);
	return node->child[cursor];
    }
    case NodeType16: {
	Node16* node=static_cast<Node16*>(n);
	return node->child[cursor];
    }
    case NodeType48: {
	Node48* node=static_cast<Node48*>(n);
	return node->child[node->childIndex[cursor]];
    }
    case NodeType256: {
	Node256* node=static_cast<Node256*>(n);
	return node->child[cursor]; 
    }
    }
    return NULL;
}

inline Node* ARTIter::nextLeaf() {
    return minimum_recordPath(nextSlot());
}

uint64_t ARTIter::value() {
    return val;
}

bool ARTIter::operator ++ (int) {
    Node* leaf = currentLeaf();
    if (index->isLeaf(leaf)) {
	val = (uint64_t)index->getLeafValue(leaf);
	nextLeaf();
	return true;
    }
    else {
	val = 0;
	return false;
    }
}

//********************************************************************
// ART
//********************************************************************

inline Node* CART::makeLeaf(uintptr_t tid) {
    // Create a pseudo-leaf
    return reinterpret_cast<Node*>((tid<<1)|1);
}

inline uintptr_t CART::getLeafValue(Node* node) {
    // The the value stored in the pseudo-leaf
    return reinterpret_cast<uintptr_t>(node)>>1;
}

//static
inline uintptr_t CART::getLeafValue(NodeStatic* node) {
    return reinterpret_cast<uintptr_t>(node)>>1;
}

inline bool CART::isLeaf(Node* node) {
    // Is the node a leaf?
    return reinterpret_cast<uintptr_t>(node)&1;
}

//static
inline bool CART::isLeaf(NodeStatic* node) {
    return reinterpret_cast<uintptr_t>(node)&1;
}

inline uint8_t CART::flipSign(uint8_t keyByte) {
    // Flip the sign bit, enables signed SSE comparison of unsigned values, used by Node16
    return keyByte^128;
}

#ifdef LOADKEY_INT
inline void CART::loadKey(uintptr_t tid,uint8_t key[]) {
    // Store the key of the tuple into the key vector
    // Implementation is database specific
    reinterpret_cast<uint64_t*>(key)[0]=__builtin_bswap64(tid);
}
#else
inline void CART::loadKey(uintptr_t tid,uint8_t key[]) {
    memcpy(reinterpret_cast<void*>(key), (const void*)tid, key_length);
}
#endif

inline unsigned CART::ctz(uint16_t x) {
    // Count trailing zeros, only defined for x>0
#ifdef __GNUC__
    return __builtin_ctz(x);
#else
    // Adapted from Hacker's Delight
    unsigned n=1;
    if ((x&0xFF)==0) {n+=8; x=x>>8;}
    if ((x&0x0F)==0) {n+=4; x=x>>4;}
    if ((x&0x03)==0) {n+=2; x=x>>2;}
    return n-(x&1);
#endif
}

inline Node** CART::findChild(Node* n,uint8_t keyByte) {
    // Find the next child for the keyByte
    switch (n->type) {
    case NodeType4: {
	Node4* node=static_cast<Node4*>(n);
	for (unsigned i=0;i<node->count;i++)
	    if (node->key[i]==keyByte)
		return &node->child[i];
	return &nullNode;
    }
    case NodeType16: {
	Node16* node=static_cast<Node16*>(n);
	__m128i cmp=_mm_cmpeq_epi8(_mm_set1_epi8(flipSign(keyByte)),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
	unsigned bitfield=_mm_movemask_epi8(cmp)&((1<<node->count)-1);
	if (bitfield)
	    return &node->child[ctz(bitfield)]; else
	    return &nullNode;
    }
    case NodeType48: {
	Node48* node=static_cast<Node48*>(n);
	if (node->childIndex[keyByte]!=emptyMarker)
	    return &node->child[node->childIndex[keyByte]]; else
	    return &nullNode;
    }
    case NodeType256: {
	Node256* node=static_cast<Node256*>(n);
	return &(node->child[keyByte]);
    }
    }
    throw; // Unreachable
}

//static
inline NodeStatic** CART::findChild(NodeStatic* n,uint8_t keyByte) {
    switch (n->type) {

    case NodeTypeD: {
	NodeD* node = static_cast<NodeD*>(n);

	if (node->count < 5) {
	    for (unsigned i = 0; i < node->count; i++)
		if (node->key()[i] == flipSign(keyByte))
		    return &node->child()[i];
	    return &nullNode_static;
	}

	for (unsigned i = 0; i < node->count; i += 16) {
	    __m128i cmp=_mm_cmpeq_epi8(_mm_set1_epi8(flipSign(keyByte)),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key(i))));
	    unsigned bitfield;
	    if (i + 16 >= node->count)
		//bitfield =_mm_movemask_epi8(cmp)&((1<<node->count)-1); THE BUGGGGGG!
		bitfield =_mm_movemask_epi8(cmp)&((1<<node->count-i)-1);
	    else
		bitfield =_mm_movemask_epi8(cmp);
	    if (bitfield)
		return &node->child(i)[ctz(bitfield)]; 
	}
	return &nullNode_static;
    }

    case NodeTypeDP: {
	NodeDP* node = static_cast<NodeDP*>(n);

	if (node->count < 5) {
	    for (unsigned i = 0; i < node->count; i++)
		if (node->key()[i] == flipSign(keyByte))
		    return &node->child()[i];
	    return &nullNode_static;
	}

	for (unsigned i = 0; i < node->count; i += 16) {
	    __m128i cmp=_mm_cmpeq_epi8(_mm_set1_epi8(flipSign(keyByte)),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key(i))));
	    unsigned bitfield;
	    if (i + 16 >= node->count)
		//bitfield =_mm_movemask_epi8(cmp)&((1<<(node->count))-1); THE BUGGGGGG!
		bitfield =_mm_movemask_epi8(cmp)&((1<<(node->count-i))-1);
	    else
		bitfield =_mm_movemask_epi8(cmp);
	    if (bitfield)
		return &node->child(i)[ctz(bitfield)]; 
	}
	return &nullNode_static;
    }

    case NodeTypeF: {
	NodeF* node = static_cast<NodeF*>(n);
	return &(node->child[keyByte]);
    }

    case NodeTypeFP: {
	NodeFP* node = static_cast<NodeFP*>(n);
	return &(node->child()[keyByte]);
    }
    }
    std::cout << "throw_static, type = " << (uint64_t)n->type <<"\n";
    throw; // Unreachable
}

inline Node* CART::minimum(Node* node) {
    // Find the leaf with smallest key
    if (!node)
	return NULL;

    if (isLeaf(node))
	return node;

    switch (node->type) {
    case NodeType4: {
	Node4* n=static_cast<Node4*>(node);
	return minimum(n->child[0]);
    }
    case NodeType16: {
	Node16* n=static_cast<Node16*>(node);
	return minimum(n->child[0]);
    }
    case NodeType48: {
	Node48* n=static_cast<Node48*>(node);
	unsigned pos=0;
	while (n->childIndex[pos]==emptyMarker)
	    pos++;
	return minimum(n->child[n->childIndex[pos]]);
    }
    case NodeType256: {
	Node256* n=static_cast<Node256*>(node);
	unsigned pos=0;
	while (!n->child[pos])
	    pos++;
	return minimum(n->child[pos]);
    }
    }
    throw; // Unreachable
}

//static
inline NodeStatic* CART::minimum(NodeStatic* node) {
    if (!node)
	return NULL;

    if (isLeaf(node))
	return node;

    switch (node->type) {
    case NodeTypeD: {
	NodeD* n=static_cast<NodeD*>(node);
	return minimum(n->child()[0]);
    }
    case NodeTypeDP: {
	NodeDP* n=static_cast<NodeDP*>(node);
	return minimum(n->child()[0]);
    }
    case NodeTypeF: {
	NodeF* n=static_cast<NodeF*>(node);
	unsigned pos=0;
	while (!n->child[pos])
	    pos++;
	return minimum(n->child[pos]);
    }
    case NodeTypeFP: {
	NodeFP* n=static_cast<NodeFP*>(node);
	unsigned pos=0;
	while (!n->child()[pos])
	    pos++;
	return minimum(n->child()[pos]);
    }
    }
    throw; // Unreachable
}


inline bool CART::leafMatches(Node* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    // Check if the key of the leaf is equal to the searched key
    if (depth!=keyLength) {
	uint8_t leafKey[maxKeyLength];
	loadKey(getLeafValue(leaf),leafKey);
	for (unsigned i=depth;i<keyLength;i++)
	    if (leafKey[i]!=key[i])
		return false;
    }
    return true;
}

//static
inline bool CART::leafMatches(NodeStatic* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    if (depth!=keyLength) {
	uint8_t leafKey[maxKeyLength];
	loadKey(getLeafValue(leaf),leafKey);
	for (unsigned i=depth;i<keyLength;i++)
	    if (leafKey[i]!=key[i])
		return false;
    }
    return true;
}

inline unsigned CART::prefixMismatch(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
    // Compare the key with the prefix of the node, return the number matching bytes
    unsigned pos;
    if (node->prefixLength>maxPrefixLength) {
	for (pos=0;pos<maxPrefixLength;pos++)
	    if (key[depth+pos]!=node->prefix[pos])
		return pos;
	uint8_t minKey[maxKeyLength];
	loadKey(getLeafValue(minimum(node)),minKey);
	for (;pos<node->prefixLength;pos++)
	    if (key[depth+pos]!=minKey[depth+pos])
		return pos;
    } else {
	for (pos=0;pos<node->prefixLength;pos++)
	    if (key[depth+pos]!=node->prefix[pos])
		return pos;
    }
    return pos;
}

//static
inline unsigned CART::prefixMismatch(NodeStatic* node,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
    switch (node->type) {
    case NodeTypeD: {
	return 0;
    }
    case NodeTypeDP: {
	NodeDP* n = static_cast<NodeDP*>(node);
	unsigned pos;
	if (n->prefixLength > maxPrefixLength) {
	    for (pos = 0; pos < maxPrefixLength; pos++)
		if (key[depth+pos] != n->prefix()[pos])
		    return pos;
	    uint8_t minKey[maxKeyLength];
	    loadKey(getLeafValue(minimum(n)),minKey);
	    for (; pos< n->prefixLength; pos++)
		if (key[depth+pos] != minKey[depth+pos])
		    return pos;
	} else {
	    for (pos = 0; pos < n->prefixLength; pos++)
		if (key[depth+pos] != n->prefix()[pos])
		    return pos;
	}
	return pos;
    }
    case NodeTypeF: {
	return 0;
    }
    case NodeTypeFP: {
	NodeFP* n = static_cast<NodeFP*>(node);
	unsigned pos;
	if (n->prefixLength > maxPrefixLength) {
	    for (pos = 0; pos < maxPrefixLength; pos++)
		if (key[depth+pos] != n->prefix()[pos])
		    return pos;
	    uint8_t minKey[maxKeyLength];
	    loadKey(getLeafValue(minimum(n)),minKey);
	    for (; pos< n->prefixLength; pos++)
		if (key[depth+pos] != minKey[depth+pos])
		    return pos;
	} else {
	    for (pos = 0; pos < n->prefixLength; pos++)
		if (key[depth+pos] != n->prefix()[pos])
		    return pos;
	}
	return pos;
    }
    }
    return 0;
}

//static
inline NodeStatic* CART::lookup(NodeStatic* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    bool skippedPrefix=false; // Did we optimistically skip some prefix without checking it?

    while (node!=NULL) {
	if (isLeaf(node)) {
	    if (!skippedPrefix && depth == keyLength) // No check required
		return node;

	    if (depth != keyLength) {
		// Check leaf
		uint8_t leafKey[maxKeyLength];
		//loadKey(getLeafValue(node), leafKey, keyLength);
		loadKey(getLeafValue(node), leafKey);
		for (unsigned i=(skippedPrefix?0:depth); i<keyLength; i++)
		    if (leafKey[i] != key[i])
			return NULL;
	    }
	    return node;
	}

	switch (node->type) {
	case NodeTypeDP: {
	    NodeDP* n = static_cast<NodeDP*>(node);
	    if (n->prefixLength < maxPrefixLength) {
		for (unsigned pos=0; pos<n->prefixLength; pos++)
		    if (key[depth+pos] != n->prefix()[pos])
			return NULL;
	    } else
		skippedPrefix=true;
	    depth += n->prefixLength;
	    break; //THE BUGGGGGG!
	}
	case NodeTypeFP: {
	    NodeFP* n = static_cast<NodeFP*>(node);
	    if (n->prefixLength < maxPrefixLength) {
		for (unsigned pos=0; pos<n->prefixLength; pos++)
		    if (key[depth+pos] != n->prefix()[pos])
			return NULL;
	    } else
		skippedPrefix=true;
	    depth += n->prefixLength;
	    break; //THE BUGGGGGG!
	}
	}

	node=*findChild(node,key[depth]);
	depth++;
    }
    return NULL;
}


inline unsigned CART::min(unsigned a,unsigned b) {
    // Helper function
    return (a<b)?a:b;
}

inline void CART::copyPrefix(Node* src,Node* dst) {
    // Helper function that copies the prefix from the source to the destination node
    dst->prefixLength=src->prefixLength;
    memcpy(dst->prefix,src->prefix,min(src->prefixLength,maxPrefixLength));
}

inline void CART::insert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength) {
    // Insert the leaf value into the tree

    if (node==NULL) {
	*nodeRef=makeLeaf(value);
	return;
    }

    if (isLeaf(node)) {
	// Replace leaf with Node4 and store both leaves in it
	uint8_t existingKey[maxKeyLength];
	loadKey(getLeafValue(node),existingKey);
	unsigned newPrefixLength=0;
	while (existingKey[depth+newPrefixLength]==key[depth+newPrefixLength])
	    newPrefixLength++;

	Node4* newNode=new Node4();
	memory += sizeof(Node4); //h
	node4_count++; //h
	newNode->prefixLength=newPrefixLength;
	memcpy(newNode->prefix,key+depth,min(newPrefixLength,maxPrefixLength));
	*nodeRef=newNode;

	insertNode4(newNode,nodeRef,existingKey[depth+newPrefixLength],node);
	insertNode4(newNode,nodeRef,key[depth+newPrefixLength],makeLeaf(value));
	num_items++; //h
	return;
    }

    // Handle prefix of inner node
    if (node->prefixLength) {
	unsigned mismatchPos=prefixMismatch(node,key,depth,maxKeyLength);
	if (mismatchPos!=node->prefixLength) {
	    // Prefix differs, create new node
	    Node4* newNode=new Node4();
	    memory += sizeof(Node4); //h
	    node4_count++; //h
	    *nodeRef=newNode;
	    newNode->prefixLength=mismatchPos;
	    memcpy(newNode->prefix,node->prefix,min(mismatchPos,maxPrefixLength));
	    // Break up prefix
	    if (node->prefixLength<maxPrefixLength) {
		insertNode4(newNode,nodeRef,node->prefix[mismatchPos],node);
		node->prefixLength-=(mismatchPos+1);
		memmove(node->prefix,node->prefix+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
	    } else {
		node->prefixLength-=(mismatchPos+1);
		uint8_t minKey[maxKeyLength];
		loadKey(getLeafValue(minimum(node)),minKey);
		insertNode4(newNode,nodeRef,minKey[depth+mismatchPos],node);
		memmove(node->prefix,minKey+depth+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
	    }
	    insertNode4(newNode,nodeRef,key[depth+mismatchPos],makeLeaf(value));
	    num_items++; //h
	    return;
	}
	depth+=node->prefixLength;
    }

    // Recurse
    Node** child=findChild(node,key[depth]);
    if (*child) {
	insert(*child,child,key,depth+1,value,maxKeyLength);
	return;
    }

    // Insert leaf into inner node
    Node* newNode=makeLeaf(value);
    switch (node->type) {
    case NodeType4: insertNode4(static_cast<Node4*>(node),nodeRef,key[depth],newNode); break;
    case NodeType16: insertNode16(static_cast<Node16*>(node),nodeRef,key[depth],newNode); break;
    case NodeType48: insertNode48(static_cast<Node48*>(node),nodeRef,key[depth],newNode); break;
    case NodeType256: insertNode256(static_cast<Node256*>(node),nodeRef,key[depth],newNode); break;
    }
    num_items++; //h
}

inline void CART::insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    if (node->count<4) {
	// Insert element
	unsigned pos;
	for (pos=0;(pos<node->count)&&(node->key[pos]<keyByte);pos++);
	memmove(node->key+pos+1,node->key+pos,node->count-pos);
	memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t));
	node->key[pos]=keyByte;
	node->child[pos]=child;
	node->count++;
    } else {
	// Grow to Node16
	Node16* newNode=new Node16();
	memory += sizeof(Node16); //h
	node16_count++; //h
	*nodeRef=newNode;
	newNode->count=4;
	copyPrefix(node,newNode);
	for (unsigned i=0;i<4;i++)
	    newNode->key[i]=flipSign(node->key[i]);
	memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
	delete node;
	memory -= sizeof(Node4); //h
	node4_count--; //h
	return insertNode16(newNode,nodeRef,keyByte,child);
    }
}

inline void CART::insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    if (node->count<16) {
	// Insert element
	uint8_t keyByteFlipped=flipSign(keyByte);
	__m128i cmp=_mm_cmplt_epi8(_mm_set1_epi8(keyByteFlipped),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
	uint16_t bitfield=_mm_movemask_epi8(cmp)&(0xFFFF>>(16-node->count));
	unsigned pos=bitfield?ctz(bitfield):node->count;
	memmove(node->key+pos+1,node->key+pos,node->count-pos);
	memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t));
	node->key[pos]=keyByteFlipped;
	node->child[pos]=child;
	node->count++;
    } else {
	// Grow to Node48
	Node48* newNode=new Node48();
	memory += sizeof(Node48); //h
	node48_count++; //h
	*nodeRef=newNode;
	memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
	for (unsigned i=0;i<node->count;i++)
	    newNode->childIndex[flipSign(node->key[i])]=i;
	copyPrefix(node,newNode);
	newNode->count=node->count;
	delete node;
	memory -= sizeof(Node16); //h
	node16_count--; //h
	return insertNode48(newNode,nodeRef,keyByte,child);
    }
}

inline void CART::insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    if (node->count<48) {
	// Insert element
	unsigned pos=node->count;
	if (node->child[pos])
	    for (pos=0;node->child[pos]!=NULL;pos++);
	node->child[pos]=child;
	node->childIndex[keyByte]=pos;
	node->count++;
    } else {
	// Grow to Node256
	Node256* newNode=new Node256();
	memory += sizeof(Node256); //h
	node256_count++; //h
	for (unsigned i=0;i<256;i++)
	    if (node->childIndex[i]!=48)
		newNode->child[i]=node->child[node->childIndex[i]];
	newNode->count=node->count;
	copyPrefix(node,newNode);
	*nodeRef=newNode;
	delete node;
	memory -= sizeof(Node48); //h
	node48_count--; //h
	return insertNode256(newNode,nodeRef,keyByte,child);
    }
}

inline void CART::insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    node->count++;
    node->child[keyByte]=child;
}


inline int CART::CompareToPrefix(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
    unsigned pos;
    if (node->prefixLength>maxPrefixLength) {
	for (pos=0;pos<maxPrefixLength;pos++) {
	    if (key[depth+pos]!=node->prefix[pos]) {
		if (key[depth+pos]>node->prefix[pos])
		    return 1;
		else
		    return -1;
	    }
	}
	uint8_t minKey[maxKeyLength];
	loadKey(getLeafValue(minimum(node)),minKey);
	for (;pos<node->prefixLength;pos++) {
	    if (key[depth+pos]!=minKey[depth+pos]) {
		if (key[depth+pos]>minKey[depth+pos])
		    return 1;
		else
		    return -1;
	    }
	}
    } else {
	for (pos=0;pos<node->prefixLength;pos++) {
	    if (key[depth+pos]!=node->prefix[pos]) {
		if (key[depth+pos]>node->prefix[pos])
		    return 1;
		else
		    return -1;
	    }
	}
    }
    return 0;
}

//static
inline int CART::CompareToPrefix(NodeStatic* n,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
    if (n->type == NodeTypeD || n->type == NodeTypeF)
	return 0;

    if (n->type == NodeTypeDP) {
	NodeDP* node = static_cast<NodeDP*>(n);
	unsigned pos;
	if (node->prefixLength>maxPrefixLength) {
	    for (pos=0;pos<maxPrefixLength;pos++) {
		if (key[depth+pos]!=node->prefix()[pos]) {
		    if (key[depth+pos]>node->prefix()[pos])
			return 1;
		    else
			return -1;
		}
	    }
	    uint8_t minKey[maxKeyLength];
	    loadKey(getLeafValue(minimum(node)),minKey);
	    for (;pos<node->prefixLength;pos++) {
		if (key[depth+pos]!=minKey[depth+pos]) {
		    if (key[depth+pos]>minKey[depth+pos])
			return 1;
		    else
			return -1;
		}
	    }
	} else {
	    for (pos=0;pos<node->prefixLength;pos++) {
		if (key[depth+pos]!=node->prefix()[pos]) {
		    if (key[depth+pos]>node->prefix()[pos])
			return 1;
		    else
			return -1;
		}
	    }
	}
    }
    else if (n->type == NodeTypeFP) {
	NodeFP* node = static_cast<NodeFP*>(n);
	unsigned pos;
	if (node->prefixLength>maxPrefixLength) {
	    for (pos=0;pos<maxPrefixLength;pos++) {
		if (key[depth+pos]!=node->prefix()[pos]) {
		    if (key[depth+pos]>node->prefix()[pos])
			return 1;
		    else
			return -1;
		}
	    }
	    uint8_t minKey[maxKeyLength];
	    loadKey(getLeafValue(minimum(node)),minKey);
	    for (;pos<node->prefixLength;pos++) {
		if (key[depth+pos]!=minKey[depth+pos]) {
		    if (key[depth+pos]>minKey[depth+pos])
			return 1;
		    else
			return -1;
		}
	    }
	} else {
	    for (pos=0;pos<node->prefixLength;pos++) {
		if (key[depth+pos]!=node->prefix()[pos]) {
		    if (key[depth+pos]>node->prefix()[pos])
			return 1;
		    else
			return -1;
		}
	    }
	}
    }
    return 0;
}

//static
inline NodeStatic* CART::findChild_recordPath(NodeStatic* n ,uint8_t keyByte, CARTIter* iter) {
    NodeStaticCursor nc;
    nc.node = n;
    switch (n->type) {
    case NodeTypeD: {
	NodeD* node=static_cast<NodeD*>(n);
	for (unsigned i=0;i<node->count;i++) {
	    //if (node->key()[i]>=keyByte) { THE BUGGGGGG!
	    //if (node->key()[i]>=flipSign(keyByte)) { THE BUGGGGGG!
	    if (flipSign(node->key()[i])>=keyByte) {
		nc.cursor = i;
		iter->node_stack.push_back(nc);
		//if (node->key()[i]==keyByte) THE BUGGGGGG!
		if (node->key()[i]==flipSign(keyByte))
		    return node->child()[i];
		else
		    return iter->minimum_recordPath(node->child()[i]);
	    }
	}
	iter->node_stack.pop_back();
	//return minimum_recordPath(nextSlot_static());
	return iter->nextLeaf();
    }
    case NodeTypeDP: {
	NodeDP* node=static_cast<NodeDP*>(n);
	for (unsigned i=0;i<node->count;i++) {
	    //if (node->key()[i]>=keyByte) { THE BUGGGGGG!
	    //if (node->key()[i]>=flipSign(keyByte)) { THE BUGGGGGG!
	    if (flipSign(node->key()[i])>=keyByte) {
		nc.cursor = i;
		iter->node_stack.push_back(nc);
		//if (node->key()[i]==keyByte) THE BUGGGGGG!
		if (node->key()[i]==flipSign(keyByte))
		    return node->child()[i];
		else
		    return iter->minimum_recordPath(node->child()[i]);
	    }
	}
	iter->node_stack.pop_back();
	//return minimum_recordPath(nextSlot_static());
	return iter->nextLeaf();
    }
    case NodeTypeF: {
	NodeF* node=static_cast<NodeF*>(n);
	if (node->child[keyByte]!=NULL) {
	    nc.cursor = keyByte;
	    iter->node_stack.push_back(nc);
	    return node->child[keyByte];
	}
	else {
	    for (unsigned i=keyByte; i<256; i++) {
		if (node->child[i]!=NULL) {
		    nc.cursor = i;
		    iter->node_stack.push_back(nc);
		    return node->child[i]; 
		}	  
	    }
	    iter->node_stack.pop_back();
	    //return minimum_recordPath(nextSlot_static());
	    return iter->nextLeaf();
	}
    }
    case NodeTypeFP: {
	NodeFP* node=static_cast<NodeFP*>(n);
	if (node->child()[keyByte]!=NULL) {
	    nc.cursor = keyByte;
	    iter->node_stack.push_back(nc);
	    return node->child()[keyByte];
	}
	else {
	    for (unsigned i=keyByte; i<256; i++) {
		if (node->child()[i]!=NULL) {
		    nc.cursor = i;
		    iter->node_stack.push_back(nc);
		    return node->child()[i]; 
		}	  
	    }
	    iter->node_stack.pop_back();
	    //return minimum_recordPath(nextSlot_static());
	    return iter->nextLeaf();
	}
    }
    }
    throw; // Unreachable
}

//static
inline NodeStatic* CART::lower_bound(NodeStatic* node, uint8_t key[], unsigned keyLength, unsigned depth, unsigned maxKeyLength, CARTIter* iter) {
    iter->node_stack.clear();
    while (node!=NULL) {
	if (isLeaf(node)) {
	    return node;
	}

	int ctp = CompareToPrefix(node,key,depth,maxKeyLength);
	if (node->type == NodeTypeDP) {
	    NodeDP* node_dp = static_cast<NodeDP*>(node);
	    depth+=node_dp->prefixLength;
	}
	else if (node->type == NodeTypeFP) {
	    NodeFP* node_fp = static_cast<NodeFP*>(node);
	    depth+=node_fp->prefixLength;
	}

	if (ctp > 0) {
	    iter->node_stack.pop_back();
	    //return minimum_recordPath(nextSlot_static());
	    return iter->nextLeaf();
	}
	else if (ctp < 0) {
	    return iter->minimum_recordPath(node);
	}

	node = findChild_recordPath(node,key[depth],iter);
	depth++;
    }
    return NULL;
}


//static
inline void CART::Node4_to_NodeD(Node4* n, NodeD* n_static) {
    for (unsigned i = 0; i < n->count; i++) {
	n_static->key()[i] = flipSign(n->key[i]);
	n_static->child()[i] = (NodeStatic*)n->child[i];
    }
}
inline void CART::Node16_to_NodeD(Node16* n, NodeD* n_static) {
    for (unsigned i = 0; i < n->count; i++) {
	n_static->key()[i] = n->key[i];
	n_static->child()[i] = (NodeStatic*)n->child[i];
    }
}
inline void CART::Node48_to_NodeD(Node48* n, NodeD* n_static) {
    unsigned c = 0;
    for (uint8_t i = 0; i < 255; i++) {
	if (n->childIndex[i] != emptyMarker) {
	    n_static->key()[c] = flipSign(i);
	    n_static->child()[c] = (NodeStatic*)n->child[n->childIndex[i]];
	    c++;
	}
    }
    if (n->childIndex[255] != emptyMarker) {
	n_static->key()[c] = flipSign(255);
	n_static->child()[c] = (NodeStatic*)n->child[n->childIndex[255]];
    }
}
inline void CART::Node256_to_NodeD(Node256* n, NodeD* n_static) {
    unsigned int c = 0;
    for (uint8_t i = 0; i < 255; i++) {
	if (n->child[i]) {
	    n_static->key()[c] = flipSign(i);
	    n_static->child()[c] = (NodeStatic*)n->child[i];
	    c++;
	}
    }
    if (n->child[255]) {
	n_static->key()[c] = flipSign(255);
	n_static->child()[c] = (NodeStatic*)n->child[255];
    }
}
inline void CART::Node_to_NodeD(Node* n, NodeD* n_static) {
    n_static->count = n->count;
    if (n->type == NodeType4)
	Node4_to_NodeD(static_cast<Node4*>(n), n_static);
    else if (n->type == NodeType16)
	Node16_to_NodeD(static_cast<Node16*>(n), n_static);
    else if (n->type == NodeType48)
	Node48_to_NodeD(static_cast<Node48*>(n), n_static);
    else if (n->type == NodeType256)
	Node256_to_NodeD(static_cast<Node256*>(n), n_static);
}

//static
inline void CART::Node4_to_NodeDP(Node4* n, NodeDP* n_static) {
    for (unsigned i = 0; i < n->count; i++) {
	n_static->key()[i] = flipSign(n->key[i]);
	n_static->child()[i] = (NodeStatic*)n->child[i];
    }
}
inline void CART::Node16_to_NodeDP(Node16* n, NodeDP* n_static) {
    for (unsigned i = 0; i < n->count; i++) {
	n_static->key()[i] = n->key[i];
	n_static->child()[i] = (NodeStatic*)n->child[i];
    }
}
inline void CART::Node48_to_NodeDP(Node48* n, NodeDP* n_static) {
    unsigned c = 0;
    for (uint8_t i = 0; i < 255; i++) {
	if (n->childIndex[i] != emptyMarker) {
	    n_static->key()[c] = flipSign(i);
	    n_static->child()[c] = (NodeStatic*)n->child[n->childIndex[i]];
	    c++;
	}
    }
    if (n->childIndex[255] != emptyMarker) {
	n_static->key()[c] = flipSign(255);
	n_static->child()[c] = (NodeStatic*)n->child[n->childIndex[255]];
    }
}
inline void CART::Node256_to_NodeDP(Node256* n, NodeDP* n_static) {
    unsigned int c = 0;
    for (uint8_t i = 0; i < 255; i++) {
	if (n->child[i]) {
	    n_static->key()[c] = flipSign(i);
	    n_static->child()[c] = (NodeStatic*)n->child[i];
	    c++;
	}
    }
    if (n->child[255]) {
	n_static->key()[c] = flipSign(255);
	n_static->child()[c] = (NodeStatic*)n->child[255];
    }
}
inline void CART::Node_to_NodeDP(Node* n, NodeDP* n_static) {
    n_static->count = n->count;
    n_static->prefixLength = n->prefixLength;
    for (unsigned i = 0; i < n->prefixLength; i++)
	n_static->prefix()[i] = n->prefix[i];

    if (n->type == NodeType4)
	Node4_to_NodeDP(static_cast<Node4*>(n), n_static);
    else if (n->type == NodeType16)
	Node16_to_NodeDP(static_cast<Node16*>(n), n_static);
    else if (n->type == NodeType48)
	Node48_to_NodeDP(static_cast<Node48*>(n), n_static);
    else if (n->type == NodeType256)
	Node256_to_NodeDP(static_cast<Node256*>(n), n_static);
}

//static
inline void CART::Node4_to_NodeF(Node4* n, NodeF* n_static) {
    for (unsigned i = 0; i < n->count; i++)
	n_static->child[n->key[i]] = (NodeStatic*)n->child[i];
}
inline void CART::Node16_to_NodeF(Node16* n, NodeF* n_static) {
    for (unsigned i = 0; i < n->count; i++)
	n_static->child[flipSign(n->key[i])] = (NodeStatic*)n->child[i];
}
inline void CART::Node48_to_NodeF(Node48* n, NodeF* n_static) {
    for (unsigned i = 0; i < 256; i++)
	if (n->childIndex[i] != emptyMarker)
	    n_static->child[i] = (NodeStatic*)n->child[n->childIndex[i]];
}
inline void CART::Node256_to_NodeF(Node256* n, NodeF* n_static) {
    for (unsigned i = 0; i < 256; i++)
	n_static->child[i] = (NodeStatic*)n->child[i];
}
inline void CART::Node_to_NodeF(Node* n, NodeF* n_static) {
    n_static->count = n->count;
    if (n->type == NodeType4)
	Node4_to_NodeF(static_cast<Node4*>(n), n_static);
    else if (n->type == NodeType16)
	Node16_to_NodeF(static_cast<Node16*>(n), n_static);
    else if (n->type == NodeType48)
	Node48_to_NodeF(static_cast<Node48*>(n), n_static);
    else if (n->type == NodeType256)
	Node256_to_NodeF(static_cast<Node256*>(n), n_static);
}

//static
inline void CART::Node4_to_NodeFP(Node4* n, NodeFP* n_static) {
    for (unsigned i = 0; i < n->count; i++)
	n_static->child()[n->key[i]] = (NodeStatic*)n->child[i];
}
inline void CART::Node16_to_NodeFP(Node16* n, NodeFP* n_static) {
    for (unsigned i = 0; i < n->count; i++)
	n_static->child()[flipSign(n->key[i])] = (NodeStatic*)n->child[i];
}
inline void CART::Node48_to_NodeFP(Node48* n, NodeFP* n_static) {
    for (unsigned i = 0; i < 256; i++)
	if (n->childIndex[i] != emptyMarker)
	    n_static->child()[i] = (NodeStatic*)n->child[n->childIndex[i]];
}
inline void CART::Node256_to_NodeFP(Node256* n, NodeFP* n_static) {
    for (unsigned i = 0; i < 256; i++)
	n_static->child()[i] = (NodeStatic*)n->child[i];
}
inline void CART::Node_to_NodeFP(Node* n, NodeFP* n_static) {
    n_static->count = n->count;
    n_static->prefixLength = n->prefixLength;
    for (unsigned i = 0; i < n->prefixLength; i++)
	n_static->prefix()[i] = n->prefix[i];
    if (n->type == NodeType4)
	Node4_to_NodeFP(static_cast<Node4*>(n), n_static);
    else if (n->type == NodeType16)
	Node16_to_NodeFP(static_cast<Node16*>(n), n_static);
    else if (n->type == NodeType48)
	Node48_to_NodeFP(static_cast<Node48*>(n), n_static);
    else if (n->type == NodeType256)
	Node256_to_NodeFP(static_cast<Node256*>(n), n_static);
}

//static
inline size_t CART::node_size(NodeStatic* n) {
    switch (n->type) {
    case NodeTypeD: {
	NodeD* node = static_cast<NodeD*>(n);
	return sizeof(NodeD) + node->count * (sizeof(uint8_t) + sizeof(NodeStatic*));
    }
    case NodeTypeDP: {
	NodeDP* node = static_cast<NodeDP*>(n);
	return sizeof(NodeDP) + node->prefixLength * sizeof(uint8_t) + node->count * (sizeof(uint8_t) + sizeof(NodeStatic*));
    }
    case NodeTypeF: {
	return sizeof(NodeF);
    }
    case NodeTypeFP: {
	NodeFP* node = static_cast<NodeFP*>(n);
	return sizeof(NodeFP) + node->prefixLength * sizeof(uint8_t) + 256 * sizeof(NodeStatic*);
    }
    }
    return 0;
}

//static
inline NodeStatic* CART::convert_to_static() {
    if (!root) return NULL;

    Node* n = root;
    NodeStatic* n_new = NULL;
    NodeStatic* n_new_parent = NULL;
    NodeStatic* returnNode = NULL;
    int parent_pos = -1;

    std::deque<Node*> node_queue;
    std::deque<NodeStatic*> new_node_queue;

    node_queue.push_back(n);
    int node_count = 0;
    while (!node_queue.empty()) {
	n = node_queue.front();
	if (!isLeaf(n)) {
	    //if ((n->count > NodeDItemTHold) || isInner(n) || (node_count < UpperLevelTHold)) {
	    if (n->count > NodeDItemTHold) {
		if (n->prefixLength) {
		    size_t size = sizeof(NodeFP) + n->prefixLength * sizeof(uint8_t) + 256 * sizeof(NodeStatic*);
		    void* ptr = malloc(size);
		    NodeFP* n_static = new(ptr) NodeFP(n->count, n->prefixLength);
		    nodeFP_count++; //h
		    Node_to_NodeFP(n, n_static);
		    n_new = n_static;
		    for (unsigned i = 0; i < 256; i++)
			if ((n_static->child()[i]) && (!isLeaf(n_static->child()[i])))
			    node_queue.push_back((Node*)n_static->child()[i]);
		}
		else {
		    size_t size = sizeof(NodeF);
		    void* ptr = malloc(size);
		    NodeF* n_static = new(ptr) NodeF(n->count);
		    nodeF_count++; //h
		    Node_to_NodeF(n, n_static);
		    n_new = n_static;
		    for (unsigned i = 0; i < 256; i++)
			if ((n_static->child[i]) && (!isLeaf(n_static->child[i])))
			    node_queue.push_back((Node*)n_static->child[i]);
		}
	    }
	    else {
		if (n->prefixLength) {
		    size_t size = sizeof(NodeDP) + n->prefixLength * sizeof(uint8_t) + n->count * (sizeof(uint8_t) + sizeof(NodeStatic*));
		    void* ptr = malloc(size);
		    NodeDP* n_static = new(ptr) NodeDP(n->count, n->prefixLength);
		    nodeDP_count++; //h
		    Node_to_NodeDP(n, n_static);
		    n_new = n_static;
		    for (unsigned i = 0; i < n_static->count; i++)
			if (!isLeaf(n_static->child()[i]))
			    node_queue.push_back((Node*)n_static->child()[i]);
		}
		else {
		    size_t size = sizeof(NodeD) + n->count * (sizeof(uint8_t) + sizeof(NodeStatic*));
		    void* ptr = malloc(size);
		    NodeD* n_static = new(ptr) NodeD(n->count);
		    nodeD_count++; //h
		    Node_to_NodeD(n, n_static);
		    n_new = n_static;
		    for (unsigned i = 0; i < n_static->count; i++)
			if (!isLeaf(n_static->child()[i]))
			    node_queue.push_back((Node*)n_static->child()[i]);
		}
	    }

	    static_memory += node_size(n_new);
	    new_node_queue.push_back(n_new);
	    node_count++;

	    bool next_parent = false;

	    if (n_new_parent) {
		if (n_new_parent->type == NodeTypeD) {
		    NodeD* node = static_cast<NodeD*>(n_new_parent);
		    node->child()[parent_pos] = n_new;
		}
		else if (n_new_parent->type == NodeTypeDP) {
		    NodeDP* node = static_cast<NodeDP*>(n_new_parent);
		    node->child()[parent_pos] = n_new;
		}
		else if (n_new_parent->type == NodeTypeF) {
		    NodeF* node = static_cast<NodeF*>(n_new_parent);
		    node->child[parent_pos] = n_new;
		}
		else if (n_new_parent->type == NodeTypeFP) {
		    NodeFP* node = static_cast<NodeFP*>(n_new_parent);
		    node->child()[parent_pos] = n_new;
		}
		else {
		    std::cout << "Node Type Error1!\t" << (uint64_t)n_new_parent->type << "\n";
		    break;
		}
	    }
	    else {
		n_new_parent = new_node_queue.front();
		returnNode = new_node_queue.front();
	    }

	    do {
		next_parent = false;

		if (n_new_parent->type == NodeTypeD) {
		    NodeD* node = static_cast<NodeD*>(n_new_parent);
		    do {
			parent_pos++;
			if (parent_pos >= node->count)
			    next_parent = true;
		    } while ((parent_pos < node->count) && isLeaf(node->child()[parent_pos]));
		}
		else if (n_new_parent->type == NodeTypeDP) {
		    NodeDP* node = static_cast<NodeDP*>(n_new_parent);
		    do {
			parent_pos++;
			if (parent_pos >= node->count)
			    next_parent = true;
		    } while ((parent_pos < node->count) && isLeaf(node->child()[parent_pos]));
		}
		else if (n_new_parent->type == NodeTypeF) {
		    NodeF* node = static_cast<NodeF*>(n_new_parent);
		    do {
			parent_pos++;
			if (parent_pos >= 256)
			    next_parent = true;
		    } while ((parent_pos < 256) && (!node->child[parent_pos] || isLeaf(node->child[parent_pos])));
		}
		else if (n_new_parent->type == NodeTypeFP) {
		    NodeFP* node = static_cast<NodeFP*>(n_new_parent);
		    do {
			parent_pos++;
			if (parent_pos >= 256)
			    next_parent = true;
		    } while ((parent_pos < 256) && (!node->child()[parent_pos] || isLeaf(node->child()[parent_pos])));
		}
		else {
		    std::cout << "Node Type Error2!\t" << (uint64_t)n_new_parent->type << "\n";
		    break;
		}

		if (next_parent) {
		    new_node_queue.pop_front();
		    if (!new_node_queue.empty())
			n_new_parent = new_node_queue.front();
		    else
			next_parent = false;
		    parent_pos = -1;
		}
	    } while (next_parent);

	    delete node_queue.front();
	    node_queue.pop_front();
	}
    }
    return returnNode;
}

inline NodeStatic* CART::convert_to_static(Node* n) {
    //if ((n->count > NodeDItemTHold) || isInner(n)) {
    if (n->count > NodeDItemTHold) {
	if (n->prefixLength) {
	    size_t size = sizeof(NodeFP) + n->prefixLength * sizeof(uint8_t) + 256 * sizeof(NodeStatic*);
	    void* ptr = malloc(size);
	    NodeFP* n_static = new(ptr) NodeFP(n->count, n->prefixLength);
	    nodeFP_count++; //h
	    Node_to_NodeFP(n, n_static);
	    for (unsigned i = 0; i < 256; i++)
		if ((n_static->child()[i]) && (!isLeaf(n_static->child()[i])))
		    n_static->child()[i] = convert_to_static((Node*)n_static->child()[i]);
	    delete n;
	    return n_static;
	}
	else {
	    size_t size = sizeof(NodeF);
	    void* ptr = malloc(size);
	    NodeF* n_static = new(ptr) NodeF(n->count);
	    nodeF_count++; //h
	    Node_to_NodeF(n, n_static);
	    for (unsigned i = 0; i < 256; i++)
		if ((n_static->child[i]) && (!isLeaf(n_static->child[i])))
		    n_static->child[i] = convert_to_static((Node*)n_static->child[i]);
	    delete n;
	    return n_static;
	}
    }
    else {
	if (n->prefixLength) {
	    size_t size = sizeof(NodeDP) + n->prefixLength * sizeof(uint8_t) + n->count * (sizeof(uint8_t) + sizeof(NodeStatic*));
	    void* ptr = malloc(size);
	    NodeDP* n_static = new(ptr) NodeDP(n->count, n->prefixLength);
	    nodeDP_count++; //h
	    Node_to_NodeDP(n, n_static);
	    for (unsigned i = 0; i < n_static->count; i++)
		if (!isLeaf(n_static->child()[i]))
		    n_static->child()[i] = convert_to_static((Node*)n_static->child()[i]);
	    delete n;
	    return n_static;
	}
	else {
	    size_t size = sizeof(NodeD) + n->count * (sizeof(uint8_t) + sizeof(NodeStatic*));
	    void* ptr = malloc(size);
	    NodeD* n_static = new(ptr) NodeD(n->count);
	    nodeD_count++; //h
	    Node_to_NodeD(n, n_static);
	    for (unsigned i = 0; i < n_static->count; i++)
		if (!isLeaf(n_static->child()[i]))
		    n_static->child()[i] = convert_to_static((Node*)n_static->child()[i]);
	    delete n;
	    return n_static;
	}
    }
}


CART::CART() 
    : root(NULL), static_root(NULL), key_length(8), memory(0), static_memory(0), num_items(0), 
      node4_count(0), node16_count(0), node48_count(0), node256_count(0), 
      nodeD_count(0), nodeDP_count(0), nodeF_count(0), nodeFP_count(0), nullNode(NULL), nullNode_static(NULL)
{}

CART::CART(unsigned kl) 
    : root(NULL), static_root(NULL), key_length(kl), memory(0), static_memory(0), num_items(0), 
      node4_count(0), node16_count(0), node48_count(0), node256_count(0), 
      nodeD_count(0), nodeDP_count(0), nodeF_count(0), nodeFP_count(0), nullNode(NULL), nullNode_static(NULL)
{}

void CART::load(vector<string> &keys, vector<uint64_t> &values, unsigned maxKeyLength) {
    uint8_t key[maxKeyLength];
    for (int i = 0; i < keys.size(); i++) {
	loadKey((uintptr_t)keys[i].c_str(), key);
	insert(key, values[i], key_length);
    }
}

void CART::load(vector<uint64_t> &keys, vector<uint64_t> &values) {
    uint8_t key[8];
    for (int i = 0; i < keys.size(); i++) {
	loadKey(keys[i], key);
	insert(key, values[i], key_length);
    }
}

void CART::insert(uint8_t key[], uintptr_t value, unsigned maxKeyLength) {
    insert(root, &root, key, 0, value, maxKeyLength);
}

void CART::convert() {
    static_root = convert_to_static();
}

uint64_t CART::lookup(uint8_t key[], unsigned keyLength, unsigned maxKeyLength) {
    NodeStatic* leaf = lookup(static_root, key, keyLength, 0, maxKeyLength);
    if (isLeaf(leaf))
	return getLeafValue(leaf);
    return (uint64_t)0;
}

uint64_t CART::lookup(uint64_t key64) {
    uint8_t key[8];
    loadKey(key64, key);
    NodeStatic* leaf = lookup(static_root, key, 8, 0, 8);
    if (isLeaf(leaf))
	return getLeafValue(leaf);
    return (uint64_t)0;
}

bool CART::lower_bound(uint8_t key[], unsigned keyLength, unsigned maxKeyLength, CARTIter* iter) {
    NodeStatic* leaf = lower_bound(static_root, key, keyLength, 0, maxKeyLength, iter);
    if (isLeaf(leaf)) {
	iter->val = (uint64_t)getLeafValue(leaf);
	iter->nextLeaf();
	return true;
    }
    else {
	iter->val = 0;
	return false;
    }
}

bool CART::lower_bound(uint64_t key64, CARTIter* iter) {
    uint8_t key[8];
    loadKey(key64, key);
    NodeStatic* leaf = lower_bound(static_root, key, 8, 0, 8, iter);
    if (isLeaf(leaf)) {
	iter->val = (uint64_t)getLeafValue(leaf);
	iter->nextLeaf();
	return true;
    }
    else {
	iter->val = 0;
	return false;
    }
}

uint64_t CART::getMemory() {
    //return memory;
    return static_memory;
}


//********************************************************************
// CARTIter
//********************************************************************
CARTIter::CARTIter() {
    index = NULL;
    val = 0;
}

CARTIter::CARTIter(CART* idx) {
    index = idx;
    val = 0;
}

inline NodeStatic* CARTIter::minimum_recordPath(NodeStatic* node) {
    if (!node)
	return NULL;

    if (index->isLeaf(node))
	return node;

    NodeStaticCursor nc;
    nc.node = node;
    nc.cursor = 0;
    node_stack.push_back(nc);

    switch (node->type) {
    case NodeTypeD: {
	NodeD* n=static_cast<NodeD*>(node);
	return minimum_recordPath(n->child()[0]);
    }
    case NodeTypeDP: {
	NodeDP* n=static_cast<NodeDP*>(node);
	return minimum_recordPath(n->child()[0]);
    }
    case NodeTypeF: {
	NodeF* n=static_cast<NodeF*>(node);
	unsigned pos=0;
	while (!n->child[pos])
	    pos++;
	node_stack.back().cursor = pos;
	return minimum_recordPath(n->child[pos]);
    }
    case NodeTypeFP: {
	NodeFP* n=static_cast<NodeFP*>(node);
	unsigned pos=0;
	while (!n->child()[pos])
	    pos++;
	node_stack.back().cursor = pos;
	return minimum_recordPath(n->child()[pos]);
    }
    }
    throw; // Unreachable
}

inline NodeStatic* CARTIter::nextSlot() {
    while (!node_stack.empty()) {
	NodeStatic* n = node_stack.back().node;
	uint16_t cursor = node_stack.back().cursor;
	cursor++;
	node_stack.back().cursor = cursor;
	switch (n->type) {
	case NodeTypeD: {
	    NodeD* node=static_cast<NodeD*>(n);
	    if (cursor < node->count)
		return node->child()[cursor];
	    break;
	}
	case NodeTypeDP: {
	    NodeDP* node=static_cast<NodeDP*>(n);
	    if (cursor < node->count)
		return node->child()[cursor];
	    break;
	}
	case NodeTypeF: {
	    NodeF* node=static_cast<NodeF*>(n);
	    for (unsigned i=cursor; i<256; i++)
		if (node->child[i]) {
		    node_stack.back().cursor = i;
		    return node->child[i]; 
		}
	    break;
	}
	case NodeTypeFP: {
	    NodeFP* node=static_cast<NodeFP*>(n);
	    for (unsigned i=cursor; i<256; i++)
		if (node->child()[i]) {
		    node_stack.back().cursor = i;
		    return node->child()[i]; 
		}
	    break;
	}
	}
	node_stack.pop_back();
    }
    return NULL;
}

inline NodeStatic* CARTIter::currentLeaf() {
    if (node_stack.size() == 0)
	return NULL;

    NodeStatic* n = node_stack.back().node;
    uint16_t cursor = node_stack.back().cursor;

    switch (n->type) {
    case NodeTypeD: {
	NodeD* node=static_cast<NodeD*>(n);
	return node->child()[cursor];
    }
    case NodeTypeDP: {
	NodeDP* node=static_cast<NodeDP*>(n);
	return node->child()[cursor];
    }
    case NodeTypeF: {
	NodeF* node=static_cast<NodeF*>(n);
	return node->child[cursor]; 
    }
    case NodeTypeFP: {
	NodeFP* node=static_cast<NodeFP*>(n);
	return node->child()[cursor]; 
    }
    }
    return NULL;
}

inline NodeStatic* CARTIter::nextLeaf() {
    return minimum_recordPath(nextSlot());
}

uint64_t CARTIter::value() {
    return val;
}

bool CARTIter::operator ++ (int) {
    NodeStatic* leaf = currentLeaf();
    if (index->isLeaf(leaf)) {
	val = (uint64_t)index->getLeafValue(leaf);
	nextLeaf();
	return true;
    }
    else {
	val = 0;
	return false;
    }
}
