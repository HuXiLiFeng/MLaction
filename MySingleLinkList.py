#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 仿照LinkList.py实现的具有相同功能的一个单向链表

class Node():
    def __init__(self,val,next=None):
        self.val=val
        self.next=next

def createList(n):
    if n<=0:
        return
    if n==1:
        return Node(1)
    else:
        head=Node(1)
        temp=head
        for i in range(2,n+1):
            temp.next=Node(i)
            temp=temp.next
    return head

def printList(head):
    p=head
    while p!=None:   #这里是判断下一个对象是否为None
        print p.val
        p=p.next

def listLength(head):
    p=head
    length=0
    while p!=None:   #这里是判断下一个对象是否为None
        length+=1
        p=p.next
    return length

def insert(head,idx,val):
    if idx<0 or idx>listLength(head):
        return
    elif idx==1:
        temp=Node(val,head)
        return temp
    else:
        p = head
        length = 1
        while length<idx-1:
            length += 1
            p = p.next

        t=Node(val)
        t.next=p.next
        p.next=t
        return head

def delete(head,idx):
    if idx<0 or idx>listLength(head):
        return
    elif idx==1:
        return head.next
    else:
        p = head
        length = 1
        while length<idx-1:
            length += 1
            p = p.next
        temp=p.next
        p.next=temp.next

        return head

if __name__=="__main__":
    a=createList(5)
    printList(a)
    print "创建链表长度为：%s"%listLength(a)
    print "=="*4
    b=insert(a,5,0)
    printList(b)
    print "=="*4
    c=delete(b,3)
    printList(c)
