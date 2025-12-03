# Product Requirements Document

## Business Problem

ACME Corp relies on a decades-old internal system to calculate travel reimbursements for employees. Built over 60 years ago, this system is still used daily despite the fact that no one fully understands how it works. The original engineers are long gone, the source code is inaccessible, and there is no formal documentation of the system's logic.

Although the system continues to operate, stakeholders have observed frequent anomalies: unpredictable reimbursement amounts, inconsistent treatment of receipts, and odd behaviors tied to specific trip lengths or distances. Attempts to document or decode the logic have failed, and different departments now hold conflicting folklore about how the system might work.

Still, the system is relied upon by Finance and HR. Replacing it is risky—but continuing to depend on an unmaintainable black box is even riskier.

8090 has built a new system but ACME Corp is confused by the differences in results. Your mission is to figure out the original business logic so we can explain why ours is different and better.

## Current Process

Employees use a legacy interface to submit:

- The number of days spent traveling
- The total number of miles traveled
- The total dollar amount of submitted receipts

The system returns a single numeric reimbursement amount with no breakdown or explanation. There is widespread belief that the result is influenced by a mix of per diem rules, mileage adjustments, receipt totals, and possibly other unknown factors.

It's also suspected that there are one or two bugs or quirks in the system's calculations—errors or artifacts from past modifications. These may produce results that appear illogical, but they are part of the current output and must be preserved in the replica.

## Project Goal

The primary goal of this project is to recreate the behavior of the legacy reimbursement system—including any known or unknown bugs that may affect output.

By replicating the current behavior exactly—warts and all—ACME can transition to a modern, maintainable codebase with confidence. Once this baseline is established, business stakeholders will be in a position to propose rule changes or improvements based on solid understanding.

## Product Description

You will build a replacement reimbursement engine that:

- Accepts the same input parameters (trip duration, miles, receipt total)
- Produces the same numeric output as the legacy system
- Matches the system's behavior across a wide variety of scenarios—including edge cases and likely bugs

To aid your reverse-engineering process, you will receive:

- 1,000 historical input/output examples
- A set of informal "discovery" interviews with long-time ACME employees

These interviews include inconsistent, anecdotal, and occasionally contradictory memories of how the system behaves

Your job is to infer the rules (or the appearance of rules) and recreate the output-producing logic as faithfully as possible.

## Requirements

- Output must match the legacy system's output with extremely high fidelity
- System must handle all 1,000 test cases with minimal or zero deviation
- Known or suspected bugs in the legacy system must be preserved in the output

## Private & Success Criteria

Your replica will be tested against the 1,000 historical reimbursement cases included in public_cases.json . The answers for these cases are provided in the file to allow you to iterate on your solution.

It will then run on 5,000 reimbursement requests in private_cases.json . The answers for these cases are not provided.

Success is defined by how closely your system's outputs match the legacy system's outputs.
