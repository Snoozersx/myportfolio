import discord
import sqlite3
from discord.ext import commands
from datetime import datetime
import asyncio

# Initialize the bot
intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

# Connect to the SQLite database
conn = sqlite3.connect('events.db')
cursor = conn.cursor()

# Create the events table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, start_time TEXT, end_time TEXT)''')
conn.commit()

# Create the attendees table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS attendees (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER, user_id INTEGER, status TEXT)''')
conn.commit()

# Create an event
@bot.command()
async def create_event(ctx, start_time: str, end_time: str):
    cursor.execute('''
    INSERT INTO events (start_time, end_time)
    VALUES (?, ?)
    ''', (start_time, end_time))
    conn.commit()
    await ctx.send(f'Event created from {start_time} to {end_time}')

# Get the current event (if there is one)
def get_current_or_future_event():
    now = datetime.now()
    cursor.execute('''
    SELECT * FROM events
    WHERE start_time >= ?
    ORDER BY start_time ASC
    ''', (now,))
    return cursor.fetchone()

@bot.command()
async def attendees(ctx):
    event = get_current_or_future_event()
    if event:
        cursor.execute('''
        SELECT user_id FROM attendees
        WHERE event_id = ?
        ''', (event[0],))
        attendees = cursor.fetchall()
        if attendees:
            attendee_list = ', '.join([str(attendee[0]) for attendee in attendees])
            await ctx.send(f'Attendees for the event starting at {event[1]}: {attendee_list}')
        else:
            await ctx.send(f'No attendees for the event starting at {event[1]}')
    else:
        await ctx.send('There is no current event')

# Check in for the current event
@bot.command()
async def checkin(ctx):
    event = get_current_or_future_event()
    if event:
        # Check if the user has already checked in
        cursor.execute('''
        SELECT * FROM attendees
        WHERE event_id = ? AND user_id = ?
        ''', (event[0], ctx.author.id))
        checkin = cursor.fetchone()
        if checkin:
            cursor.execute('''
            UPDATE attendees
            SET status = "checked in"
            WHERE id = ?
            ''', (checkin[0],))
        else:
            cursor.execute('''
            INSERT INTO attendees (event_id, user_id, status)
            VALUES (?, ?, "checked in")
            ''', (event[0], ctx.author.id))
        conn.commit()
        await ctx.send(f'{ctx.author.mention} checked in for the event starting at {event[1]}')
    else:
        await ctx.send('There is no current event')

@bot.command()
async def checkout(ctx):
    event = get_current_or_future_event()
    if event:
        # Check if the user has checked in
        cursor.execute('''
        SELECT * FROM attendees
        WHERE event_id = ? AND user_id = ?
        ''', (event[0], ctx.author.id))
        checkin = cursor.fetchone()
        if checkin:
            cursor.execute('''
            UPDATE attendees
            SET status = "checked out"
            WHERE id = ?
            ''', (checkin[0],))
            conn.commit()
            await ctx.send(f'{ctx.author.mention} checked out of the event starting at {event[1]}')
        else:
            await ctx.send(f'{ctx.author.mention} has not checked in for the event starting at {event[1]}')
    else:
        await ctx.send('There is no current event')

# Mark a user as unavailable for the current event
@bot.command()
async def unavailable(ctx):
    event = get_current_or_future_event()
    if event:
        # Check if the user has already checked in
        cursor.execute('''
        SELECT * FROM attendees
        WHERE event_id = ? AND user_id = ?
        ''', (event[0], ctx.author.id))
        attendee = cursor.fetchone()
        if attendee:
            await ctx.send(f'You have already checked in for the event starting at {event[1]}')
        else:
            cursor.execute('''
            INSERT INTO attendees (event_id, user_id, status)
            VALUES (?, ?, 'unavailable')
            ''', (event[0], ctx.author.id))
            conn.commit()
            await ctx.send(f'You have marked yourself as unavailable for the event starting at {event[1]}')
    else:
        await ctx.send('There are no upcoming events')

# Remove old events
def remove_old_events():
    now = datetime.now()
    cursor.execute('''
    DELETE FROM events
    WHERE end_time <= ?
    ''', (now,))
    conn.commit()

async def remove_old_events_task():
    while True:
        remove_old_events()
        await asyncio.sleep(3600) # Sleep for 1 hour

# Start the background task
bot.loop.create_task(remove_old_events_task())
# Start the bot
bot.run('YOUR_DISCORD_BOT_TOKEN')
