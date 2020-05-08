/** Keep in mind that aggregate function calls cannot be nested **/

-- Count total numbebr of transactions and total spendings for each card holder
select count(t.id) as Transaction_Count, round(cast(sum(t.amount) as numeric), 3) as Total_Spending, ch.id as "Card Holder ID", ch.name as "Card Holder Name"
from transaction t 
join credit_card cc
on t.card_number = cc.card_number
join card_holder ch
on cc.id_holder = ch.id
group by cc.card_number, ch.id, ch.name
order by Transaction_Count desc;



-- Top 100 highest transactions amount during 7am and 9am
select t.date, t.amount, ch.id, ch.name
from transaction t 
join credit_card cc
on t.card_number = cc.card_number
join card_holder ch
on cc.id_holder = ch.id
where date_part('hour',t.date) between 7 and 8
--in (7,8)
order by t.amount desc
limit 100;



-- Top 5 merchants prone to being hacked using small transactions
select count(t.id), m.id, m.name
from transaction t
join merchant m
on t.id_merchant = m.id
where t.amount < 2
group by m.id
order by count(t.id) desc
limit 5;



/** Section 1 **/
-- Count the transactions that are less than $2 for each card holder
select count(t.id), ch.id, ch.name
from transaction t
join credit_card cc
on t.card_number = cc.card_number
join card_holder ch
on cc.id_holder = ch.id
where t.amount > 2
group by ch.id
order by count(t.id) desc



-- Get the highest count of the small transactions of < $2
-- The highest count is 202 times
select max(a.count_small_tran) from
	(select count(t.id) as count_small_tran, ch.id as cust_id
	from transaction t
	join credit_card cc
	on t.card_number = cc.card_number
	join card_holder ch
	on cc.id_holder = ch.id
	where t.amount > 2
	group by ch.id) as a;
	
	
	
-- Show the name and id of the person who has 202 transactions of < $2
-- The person is Crystal Clark
select count(t.id), ch.id, ch.name
from transaction t
join credit_card cc
on t.card_number = cc.card_number
join card_holder ch
on cc.id_holder = ch.id
where t.amount > 2
group by ch.id, ch.name
having count(t.id) = (
	select max(a.count_small_tran) from
		(select count(t.id) as count_small_tran, ch.id as cust_id
		from transaction t
		join credit_card cc
		on t.card_number = cc.card_number
		join card_holder ch
		on cc.id_holder = ch.id
		where t.amount > 2
		group by ch.id) as a
);



-- Return the credit card numbers, credit card id and holder 
-- who has the highest number of transaction that are less than $2
select cc.card_number, ch.id, ch.name
from credit_card cc
join card_holder ch
on cc.id_holder = ch.id
where ch.id = (
	select ch.id
	from transaction t
	join credit_card cc
	on t.card_number = cc.card_number
	join card_holder ch
	on cc.id_holder = ch.id
	where t.amount > 2
	group by ch.id, ch.name
	having count(t.id) = (
		select max(a.count_small_tran) from
			(select count(t.id) as count_small_tran, ch.id as cust_id
			from transaction t
			join credit_card cc
			on t.card_number = cc.card_number
			join card_holder ch
			on cc.id_holder = ch.id
			where t.amount > 2
			group by ch.id) as a
	)
);



-- Create a view of top 5 people who have the highest count of small transactions
-- and their credit card numbers
create view top_5_small_tran_card_holders as (
	select cc.card_number, ch.id, ch.name
	from credit_card cc
	join card_holder ch
	on cc.id_holder = ch.id
	where ch.id in (
		select ch.id
		from transaction t
		join credit_card cc
		on t.card_number = cc.card_number
		join card_holder ch
		on cc.id_holder = ch.id
		where t.amount > 2
		group by ch.id, ch.name
		having count(t.id) in (
			select count(t.id) as count_small_tran
			from transaction t
			join credit_card cc
			on t.card_number = cc.card_number
			join card_holder ch
			on cc.id_holder = ch.id
			where t.amount > 2
			group by ch.id
			order by count_small_tran desc
			limit 5		
		)
	)
)
;



-- select from view top_5_small_tran_card_holders
select * from top_5_small_tran_card_holders;











