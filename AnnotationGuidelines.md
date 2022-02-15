We followed guidelines from the I2B2 2014 de-identification challenge (Stubbs and Uzuner, JBI 2015).\
We added/refined definitions for some entity types based on frequently occurring patterns in the MGB clinical notes 

### Prodigy Setup

create a conda environment, install spacy and a few pretrained spacy models.

```bash
conda env create --file prodigy.yml
conda activate prodigy

python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

install prodigy
```bash
pip install prodigy.X.Y.Z.whl
```

### Run prodigy to collect annotations

There are different modes of annotating data, we should test and see which one is most appropriate for us. 
We used the fully manual annotation on text that is pre-labeled using a model in the loop.

```
prodigy ner.manual ner_news_headlines en_core_web_lg ./data/news_headlines.jsonl --label PERSON,ORG,PRODUCT,LOCATION --highlight-chars
```

### PHI Tags/Labels

HIPAA protected tag types (PHI Categories) are mentioned in **bold**.

1. **PATIENT** - Refers to the name of the patient or any other entity (non-medical) associated with the patient (e.g. wife's name, husband's name, daughter's name etc.).
2. STAFF - STAFF is used as an umbrella term for all hospital staff, including nurses, pharmacists, receptionists. Includes abbreviations (AB123=Andrew Bear, MD).
3. **AGE** - Annotate all ages, including those for patient’s families or those within the past history list. Tag the month/week age if written.
4. **DATE** - Any calendar date, including years, seasons, months, and holidays, days of the week, should be annotated. Time should not be annotated.
5. **PHONE**: Any of the following contact details: **PHONE** and **FAX**.
6. **ID**: Includes any personal identification numbers, but not limited to - **MRN** (Medical Record Number), **USERNAME**, **SSN**, **HEALTHPLAN**, **ACCOUNT ID**, **LICENSE NUMBER**, **VEHICLE ID**, **DEVICE ID**, **BIOID**, **IDNUM** (IDNUM refers to any numbers that don't match the other ID categories)
7. **EMAIL**: Any email address.
8. **PATORG**: Refers to the organization entity associated with the patient (not a hospital entity - e.g. works at [Google]. Lawyer at [Harvard]).
9. LOC: Refers to location & includes - **STREET**, **CITY**, STATE, COUNTRY, **ZIP**. The address of the hospital goes into this category.
10. HOSP: Refers to hospital related information & includes - hospital names, pharmacy names and direct hospital identifier numbers.
11. OTHERPHI: Refers to a catch-all for information that could not be classified as any other PHI, but that could still potentially provide information about the patient
12. OTHERISSUE: Refers to anything broken in a note that hinders the annotation process (e.g - Merged words that can’t be annotated). This would probably arise because of tokenization issues etc.

### Deciding tricky cases

If we're having trouble trying to decide which tag to use or what to annotate and what not to annotate, it can help to "simulate" making another pseudonymous note:
```
Original note -> De-identified ->  assign random values based on tags - PATIENT, HOSP, LOC -> Pseudonymous Note
```

## Detailed guidelines

### GENERAL

* In order to de-identify the records, each file must have the PHI marked up so that it can be removed/replaced later.
* This will be done using a graphical interface - **Prodigy**
* When tagging something that is PHI, but it’s not obvious what to tag it as, think about what it should be replaced by, and whether that will make sense in the document.

### PATIENT

* Name of the patient or any other entity (non-medical) associated with the patient (e.g. wife, husband, daughter, care partners etc.).
* Annotate all care partners & healthcare proxies as PATIENT
* If unsure whether it is a PATIENT name or STAFF name, tag it as PATIENT - always the more strict PHI category.
* Titles (Dr., Mr., Ms., etc.) do not have to be annotated.
* If a name is possessive (e.g., Sam’s) do not annotate the ’s
* The span should include the entire PATIENT name. Don't split the same name into multiple spans.
* Make sure two different patients are not part of the same span.
* Include the period (in the span) present after the single initial - e.g. John Doe D. M.S. - the period after D is included in the span - **[John Doe D.]**<sub>PATIENT</sub> .

#### Example

_Original Text:_

Patient: Bruce Wayne received his suit in the morning and Mr. Norris, Chuck K received his flu shot today. They were treated by Apollo Creed M.D at the Mayfield Psychiatric Hospital where Alfred,Pennyworth J from 2014 Gotham Street was treated by Physician: 
Gregory House M.D

_De-Identified Text:_

Patient: **[Bruce Wayne]**<sub>PATIENT</sub> received his suit in the morning and Mr. **[Norris, Chuck K]**<sub>PATIENT</sub> received his flu shot today. They were treated by **[Apollo Creed]**<sub>STAFF</sub> M.D at the **[Mayfield Psychiatric Hospital]**<sub>HOSP</sub> where **[Alfred,Pennyworth J]**<sub>PATIENT</sub> from **[2014 Gotham Street]**<sub>LOC</sub> was treated by Physician: **[Gregory House]**<sub>STAFF</sub> M.D

### STAFF

* STAFF is used as an umbrella term for all hospital staff, including nurses, pharmacists, receptionists, and so on
* Annotate initials of the STAFF member, or some variation of their name, used for identification purposes - e.g. MB31. These initials are usually seen around the STAFF name or around test results sometimes.
* Titles (Dr., Mr., Ms., etc.) do not have to be annotated.
* Information such as “M.D.”, “R.N.” do not have to be annotated
* If a name is possessive (e.g., Sam’s) do not annotate the ’s
* The span should include the entire STAFF name. In some medical notes, the tokens M.D or R.N etc. might occur in between the STAFF name. This should not be included in the span. In this case this we would have one span on the left and another span on the right. E.g. John M.D Doe will be annotated as **[John]**<sub>STAFF</sub> M.D **[Doe]**<sub>STAFF</sub>

#### Example

_Original Text:_

Mr. Rocky Balboa was treated by Apollo Creed and there was a follow-up by James E. Wilson (JW17) M.D, Oncology and the patient BP 112/80 JW17 was recorded.

_De-Identified Text:_

Mr. **[Rocky Balboa]**<sub>PATIENT</sub> was treated by **[Apollo Creed]**<sub>STAFF</sub> and there was a follow-up by **[James E. Wilson]**<sub>STAFF</sub> (**[JW17]**<sub>STAFF</sub>) M.D, **[Oncology]**<sub>HOSP</sub> and the patient BP 112/80 **[JW17]**<sub>STAFF</sub> was recorded.

### AGE

* Annotate all ages, including those for patient’s families or those within the past history list. Include also the month/week if written.
* If the AGE has text associated with it (unless the text corresponds to the words related to week/month), that is not split by whitespace (e.g - 70yo, 4y, 35M - M refers to Male, 72F - F refers to Female), the text will not be included in the annotation.
* If an AGE is mentioned, but it is not associated with to the patient (e.g. refers to a medical test statistic), then you do not need to annotate that as AGE.
* If the age is an approximation - like he got the virus in his 80s - the letter "s" will be included in the annotation. If there is an apostrophe - then it is also included - 80's - the apostrophe and "s" will be included in the annotation.

#### Example

_Original Text:_

The patient is a 72yo woman who got tested in her mid 20s. She had a test done in her early 30's. Selina Kyle a 63F came to BWH for treatment by Dr Nick Fury. She was accompanied by her 18 year 4m son.

_De-Identified Text:_

The patient is a **[72]**<sub>AGE</sub>yo woman who got tested in her mid **[20s]**<sub>AGE</sub>. She had a test done in her early **[30's]**<sub>AGE</sub>. Selina Kyle a **[63]**<sub>AGE</sub>F came to **[BWH]**<sub>HOSP</sub> for treatment by Dr **[Nick Fury]**<sub>STAFF</sub>. She was accompanied by her **[18]**<sub>AGE</sub> year **[4m]**<sub>AGE</sub> son.

### DATE

* Any calendar date, including years, seasons (spring, fall), months, and holidays, days of the week, should be annotated. Time need not be annotated.
* Days of the week should also be annotated
* Do not include time of day
* If the phrase has ’s (i.e., in the ’90s), annotate **’90s** - the apostrophe **’** is included.
* Include annotations of seasons ("Fall ’02")
* Include quote marks that stand in for years e.g - **’92** - the apostrophe **’** is included.
* Date ranges will be annotated as two separate spans (e.g. 2004-2005 - 2004 and 2005 will be annotated as two separate DATE spans)
* Sometimes certain tokens will look like dates, but might not be dates and could be the value of a test result (e.g. Apgars 8/9)
* Years could look like time (e.g. 2000 - could refer to the time being 8PM)

#### Example

_Original Text:_

The date: 03/03/21 and DISCHARGE PATIENT:07/05/00 12:00PM. The patient reports falling sick between the years 2004-2005 and also in Fall 2006. She told us on Tuesday that she had another test done on Jan, 23rd 2050. TESTS: Apgars 8/9 BP 110/120 Systolic Murmur 1/6. These were reported at 03/03/21 2000. Take the medicine on M W F.

_De-Identified Text:_

The date: **[03/03/21]**<sub>DATE</sub> and DISCHARGE PATIENT:**[07/05/00]**<sub>DATE</sub> 12:00PM. The patient reports falling sick between the years **[2004]**<sub>DATE</sub>-**[2005]**<sub>DATE</sub> and also in **[Fall 2006]**<sub>DATE</sub>. She told us on **[Tuesday]**<sub>DATE</sub> that she had another test done on **[Jan, 23rd 2050]**<sub>DATE</sub>. TESTS: Apgars 8/9 BP 110/120 Systolic Murmur 1/6. These were reported at **[03/03/21]**<sub>DATE</sub> 2000. Take the medicine on **[M]**<sub>DATE</sub> **[W]**<sub>DATE</sub> **[F]**<sub>DATE</sub>.

### PHONE

* This would include all phone, fax and pager numbers, irrespective of whom it belongs to (e.g. patient, doctor etc.).
* If the phone number has brackets, the brackets will be part of the span. Text will not be part of the phone number.
* Generally pager numbers are preceded by small (x) or capital (X) or small (p) or the word (pager) and the pager numbers are usually 5 numbers long.

#### Example

_Original Text:_

The doctor can be reached at X1-1234 or at her phone +1 619-911-619. You can page the nurse: pager 12345 or the pharmacist p12345
or the therapist at X12345. The contact for the hospital is 800-273-8255

_De-Identified Text:_

The doctor can be reached at X **[1-1234]**<sub>PHONE</sub> or at her phone **[+1 619-911-619]**<sub>PHONE</sub>. You can page the nurse: pager **[12345]**<sub>PHONE</sub> or the pharmacist p **[12345]**<sub>PHONE</sub> or the therapist at X **[12345]**<sub>PHONE</sub>. The contact for the hospital is **[800-273-8255]**<sub>PHONE</sub>

### ID
* Includes - medical record numbers, usernames, SSN, healthplan numbers, account details, license details, vehicle details, device ID, bio id and IDNUM (IDNUM refers to any numbers that don't match the other ID categories)
* When in doubt, annotate it as ID
* Provider Number, Unit #, Unit No, Account Number, ACCT # etc. are all ID
* P1637373 - 1637373 is an ID (although be careful because when there is a small "p" followed by 5 digits, it's usually a pager number - p12345 - where 12345 is a pager number)
* No need to label names of devices (for example: “25 mm Carpentier-Edwards magna valve”, “3.5 mm by 32 mm Taxus drug-eluting stent”, Angioseal”)
* ICD-10, CPT and HCPCS codes are non PHI, so they should not be labelled as ID

#### Example

_Original Text:_

Patient: Jack MRN: 1123443334. Wife Jill 23453223 BWH. Son: JJ \[12345678BWH]. They all received treatment yesterday. Unit No: 123987, Account Number: 345678. SSN:333-22-4444.

_De-Identified Text:_

Patient: **[Jack]**<sub>PATIENT</sub> MRN: **[1123443334]**<sub>ID</sub>. Wife **[Jill]**<sub>PATIENT</sub> **[23453223]**<sub>ID</sub> **[BWH]**<sub>HOSP</sub>. Son **[JJ]**<sub>PATIENT</sub> \[**[12345678]**<sub>ID</sub>**[BWH]**<sub>HOSP</sub>]. hey all received treatment yesterday and went up the hill. Unit No: **[123987]**<sub>ID</sub>, Account Number: **[345678]**<sub>ID</sub> and SSN: **[333-22-4444]**<sub>ID</sub>.

### EMAIL

* Any e-mail address mentioned in the text.

#### Example

_Original Text:_

Mail the test results to gordon\@gotham.com and reach out to the doctor at sundar.pichai\@gmail.com or at house@partners.org.

_De-Identified Text:_

Mail the test results to **[gordon\@gotham.com]**<sub>EMAIL</sub> and reach out to the doctor at **[sundar.pichai\@gmail.com]**<sub>EMAIL</sub> or at **[house@partners.org]**<sub>EMAIL</sub>.

### PATORG

* Refers to the organization entity associated with the patient (not a hospital entity - e.g. works at [Google]. Lawyer at [Harvard])
* Any organization that is mentioned that is not associated with someone on the medical staff should be tagged
* Profession need not be tagged.

#### Example

_Original Text:_

Bruce Wayne owns Wayne Enterprises. His nemesis Mr. Clark Kent, a journalist at the daily planet was treated by Dr. House.

_De-Identified Text:_

**[Bruce Wayne]**<sub>PATIENT</sub> owns **[Wayne Enterprises]**<sub>PATORG</sub>. His nemesis Mr. **[Clark Kent]**<sub>PATIENT</sub> , a journalist at the **[daily planet]**<sub>PATORG</sub> was treated by Dr. **[House]**<sub>STAFF</sub>

### LOC

* Stands for location.
* Annotate state/country names as well as addresses and cities. Annotate street, city, state, zip and country as separate entities.
* Generic locations like “hair salon” do not need to be annotated, but named organizations (i.e., “Harvard University”) do
* If in doubt, annotate

#### Example

_Original Text:_

Bruce Wayne was treated at Head and Neck Oncology Center, Dana-Farber Cancer Institute,  450 Brookline Ave, Boston, MA, 02215. The test results were presented to Mrs. Selina Kyle. The results from BWH will be sent to 1007 Mountain Drive, Gotham, NJ. Their friend Mr. Aquaman works has part time job at the salon and also works at One Brave Idea.

_De-Identified Text:_

**[Bruce Wayne]**<sub>PATIENT</sub>  was treated at Head and Neck Oncology Center, **[Dana-Farber Cancer Institute]**<sub>HOSP</sub>,  **[450 Brookline Ave]**<sub>LOC</sub>, **[Boston]**<sub>LOC</sub>, **[MA]**<sub>LOC</sub>, **[02215]**<sub>LOC</sub>. The test results were presented to Mrs. **[Selina Kyle]**<sub>PATIENT</sub>. The results from **[BWH]**<sub>HOSP</sub> will be sent to **[1007 Mountain Drive]**<sub>LOC</sub>, **[Gotham]**<sub>LOC</sub>, **[NJ]**<sub>LOC</sub>.

### HOSP
* Refers to hospital names and rooms.
* Floor, suite and room numbers will be annotated as hospital - e.g - [Floor 2, room 254]<sub>HOSP</sub>
* Generic departments are not annotated (e.g. Department of Cardiology will not be annotated)

_Original Text:_

Bruce Wayne was treated at Head and Neck Oncology Center, Dana-Farber Cancer Institute,  450 Brookline Ave, Boston, MA, 02215. The patient was treated at Bigelow room C. The results from BWH, floor: floor 5 will be sent to 1007 Mountain Drive, Gotham, NJ.

_De-Identified Text:_

**[Bruce Wayne]**<sub>PATIENT</sub>  was treated at Head and Neck Oncology Center, **[Dana-Farber Cancer Institute]**<sub>HOSP</sub>,  **[450 Brookline Ave]**<sub>LOC</sub>, **[Boston]**<sub>LOC</sub>, **[MA]**<sub>LOC</sub>, **[02215]**<sub>LOC</sub>. The patient was treated at **[Bigelow room C]**<sub>HOSP</sub>. The results from **[BWH]**<sub>HOSP</sub>, floor: **[floor 5]**<sub>HOSP</sub> will be sent to **[1007 Mountain Drive]**<sub>LOC</sub>, **[Gotham]**<sub>LOC</sub>, **[NJ]**<sub>LOC</sub>.

### OTHERPHI

* Refers to a catch-all for information that could not be classified as any other PHI, but that could still potentially provide information about the patient
* For example, the description of a patient’s injuries as “resulting from Superstorm Sandy” would not be covered under the HIPAA guidelines, but they indirectly provide both a location and a year for that medical record. This information, paired with other hints about the patient’s identity, such as profession and number of children, could lead to the patient’s identity.
* Another example that could potentially provide information about the patient would be something like “is excited to see the Red Sox play a home game in the World Series next week”.

### OTHERISSUE

* Refers to anything broken that’s not PHI (e.g - Merged words that can’t be annotated). This would probably arise because of tokenization issues etc. Should hopefully be corrected while performing annotations. 
* This tag will be used when you want to annotate something as PHI, but you are unable to exactly select the span (of PHI) that you want to annotate. Maybe there are some extra words or characters being included (because of some tokenization issues or processing on our end).
* Issues as mentioned above will be marked as OTHERPHI


## Thoughts

If we're having trouble trying to decide which tag to use or what to annotate and what not to annotate, one way to think about it would be:

Note ---> De-identified ---> Re-identified (assigning random values based on tags - HOSP, LOC etc.) ---> Pseudo Note

Once we de-identify a note and wanted to re-identify it - what process (e.g. which tags) would still keep it (the pseudo note) semantically relevant to the original note. Taking a look the examples might give us a better idea. 

### Example 1

Say we have the following note:

..... Head and Neck Oncology Center, Dana-Farber Cancer Institute, 450 Brookline Ave, Boston MA 02215 .....

**Approach 1:**

*De-identified:*

..... Head and Neck Oncology Center, HOSP[Dana-Farber Cancer Institute], LOC[450 Brookline Ave], LOC[Boston] LOC[MA] LOC[02215] .....

*Re-identified*:

The de-identified note would look like this: 

..... Head and Neck Oncology Center, [HOSP] [LOC], [LOC] [LOC] [LOC] .....

If we were looking to re-identify this note with some pseudo values (random values), we would pick a random hospital name (HOSP - e.g. BWH) and location (LOC - e.g. 360 Huntington Ave, Boston, MA, 02120)

And now the re-identified text would be: 

..... Head and Neck Oncology Center, BWH, 360 Huntington Ave, Boston MA 02120 .....

**Approach 2:**

*De-identified:*

..... LOC[Head and Neck Oncology Center, Dana-Farber Cancer Institute, 450 Brookline Ave Boston MA 02215] ......

*Re-identified*:

The de-identified note would look like this: 

..... [LOC] .....

If I was looking to re-identify this note with some pseudo values (random values), we would pick a random location (LOC - e.g. 360 Huntington Ave, Boston, MA)

And now the re-identified text would be: 

..... 360 Huntington Ave, Boston, MA .....

In this example taking **Approach 1** would be better than taking **Approach 2** , since the re-identified text in **Approach 1** bears more resemblance (semantically) to the original text.

### Example 2

Say we have the following note:

..... treated at the 360 Longwood clinic .....

**Approach 1:**

*De-identified:*

..... treated at the LOC[360 Longwood clinic] .....

*Re-identified*:

The de-identified note would look like this: 

..... treated at the [LOC] .....

If we were looking to re-identify this note with some pseudo values (random values), we would pick a random location (LOC - e.g. 360 Huntington Ave)

And now the re-identified text would be: 

..... treated at the 360 Huntington Ave, Boston, MA .....

**Approach 2:**

*De-identified:*

..... treated at the HOSP[360 Longwood clinic] .....

*Re-identified*:

The de-identified note would look like this: 

..... treated at the [HOSP] .....

If we were looking to re-identify this note with some pseudo values (random values), we would pick a random hospital name (HOSP - e.g. BWH)

And now the re-identified text would be: 

..... treated at the BWH .....

In this example taking **Approach 2** would be better than taking **Approach 1** , since the re-identified text in **Approach 2** bears more resemblance (semantically) to the original text.

## HIPAA

* PHI categories defined by HIPAA

1. Names

2. All geographic subdivisions smaller than a State, including street address, city, county, precinct, zip code, and their equivalent geocodes, except for the initial three digits of a zip code if, according to the current publicly-available data from the Bureau of the Census:

    (a) The geographic unit formed by combining all zip codes with the same three initial digits contains more than 20,000 people; and

    (b) The initial three digits of a zip code for all such geographic units containing 20,000 or fewer people is changed to 000.

3. All elements of dates (except year) for dates directly related to an individual, including birth date, admission date, discharge date, date of death; and all ages over 89 and all elements of dates (including year) indicative of such age, except that such ages and elements may be aggregated into a single category of age 90 or older

4. Telephone numbers

5. Fax numbers

6. Electronic mail addresses

7. Social security numbers

8. Medical record numbers

9. Health plan beneficiary numbers

10. Account numbers

11. Certificate/license numbers

12. Vehicle identifiers and serial numbers, including license plate numbers

13. Device identifiers and serial numbers

14. Web Universal Resource Locators (URLs)

15. Internet Protocol (IP) address numbers

16. Biometric identifiers, including finger and voice prints

17. Full face photographic images and any comparable images

18. Any other unique identifying number, characteristic, or code.
