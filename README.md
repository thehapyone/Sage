# CodeSage - AI Agent for your Workplace


## What is CodeSage
The ultimate objective of sage is to a full fledge AGI that will be capable of working fully autonomosoly in software development capacity.

For the time being, here the main objectives:
 - To assist users in analysing, generating, debugging, performing static and dynamic analysi. Fault detection, code documentaiton, CI/CD evaluation and many more
 - To reduce the need for human reviewers as much as possible
 - To assist in resolving tickets in Jira and any other Issue management system :> i.e act as a jira action agent

## Sage Core

### Objective

### Response

### Behaviour
Sage should:
 - Behave in a professional and courteus manner, offering insightful and constructive feedback to help users inprove their coding skills
 - Should be able to identity potential bugs, errors, performance issues, and other code quality concerns
 - Should be customizable and adaptable as much as possible to the human needs


## Modes
The first iteration of sage will be capable of three main modes:
  - Sage Chat
  - Sage Reviewer
  - Sage Jira

## Sage Reviewer
This is the default deploymet mode of sage. In this mode, sage attempts to act as an expert code guru capable of replacing/suplemnting human reviewer for code changes.

It will be capable of analying code changes and suggest various recommendations based on the changes and configuration.

## Sage Chat

This is a complimentary mode. Here sage is capabl eof having a natural chat conversaion with the users. This could be related to code, data sources, and many more. The core of the chat mode is Retrieval Augmented Generation (RAG).

### What is RAG?
RAG is a technique for augmenting LLM knowledge with additional data.

### RAG Architecture
A typical RAG application has two main components:

**Indexing**: a pipeline for ingesting data from a source and indexing it. This usually happens offline.

**Retrieval and generation**: the actual RAG chain, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

## Sage Agent
In agent mode, sage acts as an autonomous AI agent able to perform some specific set of actions for example:
 - Create a MR to fix a bug
 - As a Jira Agent
   - Here, sage can be assign a ticket in Jira and then it analyze the ticket content and then decide a set of actions that could be used to address the ticket - When given the go ahead, it will proceed to execute the actions.

