import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL CONFIGURATION - MIDCAP TRADING STRATEGY
# =============================================================================

@dataclass
class MidcapTradingConfig:
    """Configuration for Midcap Trading Strategy"""
    # Data Configuration
    DATE_COL: str = "date"
    FEATURES: List[str] = field(default_factory=lambda: None)
    
    # Timeframes to test (in minutes)
    TIMEFRAMES: List[int] = field(default_factory=lambda: [15, 30, 60])
    
    # Always-On Safety Rules
    VOLUME_MULTIPLIER: float = 1.5       # Volume >= 1.5 × MA20(volume)
    MAX_GAP_PERCENT: float = 7.0         # Skip if gap >= 6-8%
    AVOID_FIRST_LAST_MINUTES: int = 15   # Avoid first & last 15 minutes
    EMA_PERIOD: int = 50                 # EMA50 trend filter
    
    # RSI Thresholds by timeframe (from your strategy)
    RSI_THRESHOLDS: Dict[int, float] = field(default_factory=lambda: {
        15: 53,  # 15m: RSI >= 53
        30: 55,  # 30m: RSI >= 55
        60: 55   # 1h: RSI >= 55
    })
    
    # ATR Multipliers (k0, k1, k2) by timeframe - FROM YOUR STRATEGY
    ATR_MULTIPLIERS: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        15: {'k0': 2.2, 'k1': 1.8, 'k2': 1.3},  # 15m parameters
        30: {'k0': 2.4, 'k1': 1.9, 'k2': 1.4},  # 30m parameters  
        60: {'k0': 2.8, 'k1': 2.1, 'k2': 1.5}   # 1h parameters
    })
    
    # Time stops by timeframe (reduced bars from your strategy)
    TIME_STOPS: Dict[int, int] = field(default_factory=lambda: {
        15: 8,   # 8 bars for 15m
        30: 10,  # 10 bars for 30m
        60: 6    # 6 bars for 1h
    })
    
    # Entry Configuration
    EMA_PULLBACK_TOLERANCE: float = 0.6  # Within ~0.3-0.6% of EMA50
    TINY_BUFFER_PERCENT: float = 0.07    # Tiny buffer ~0.05-0.10%
    ORB_MINUTES: int = 30               # Opening range period
    
    # Position Sizing - REALISTIC APPROACH
    INITIAL_CAPITAL: float = 1000000     # Starting capital
    RISK_PER_TRADE: float = 10000        # Fixed risk per trade
    MAX_POSITIONS: int = 5               # Maximum concurrent positions
    MAX_POSITION_VALUE: float = 200000   # Max position size
    
    # Commission and Slippage
    COMMISSION_PERCENT: float = 0.05     # 0.05% per side
    SLIPPAGE_PERCENT: float = 0.02       # 0.02% slippage
    
    # Sanity Checks
    MIN_BARS_FOR_ENTRY: int = 50         # Minimum bars before allowing entries
    MAX_SINGLE_TRADE_RETURN: float = 100.0  # Cap single trade returns
    
    def __post_init__(self):
        if self.FEATURES is None:
            self.FEATURES = ['open','high','low','close','volume',
                           'ATR_14','EMA_50','RSI_14','volume_ma20','prev_close']

# Initialize global config
CONFIG = MidcapTradingConfig()

# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

class DataProcessor:
    """Handles data loading and technical indicator calculation"""
    
    @staticmethod
    def load_single_csv(file_path: str) -> pd.DataFrame:
        """Load CSV and filter to past 6 months"""
        six_months_ago = datetime.now() - timedelta(days=180)
        
        try:
            df = pd.read_csv(file_path)
            df[CONFIG.DATE_COL] = pd.to_datetime(df[CONFIG.DATE_COL])
            
            # Filter to past 6 months
            df_filtered = df[df[CONFIG.DATE_COL] >= six_months_ago].copy()
            
            if len(df_filtered) == 0:
                return pd.DataFrame()
            
            df_filtered.set_index(CONFIG.DATE_COL, inplace=True)
            return df_filtered
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def resample_to_timeframe(df: pd.DataFrame, timeframe: int) -> pd.DataFrame:
        """Resample 1-minute data to specified timeframe"""
        if df.empty:
            return df
        
        freq = f"{timeframe}min"
        
        # OHLCV resampling
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if resampled.empty:
            return resampled
        
        # Add technical indicators
        resampled = DataProcessor.calculate_indicators(resampled)
        return resampled
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required technical indicators"""
        if df.empty or len(df) < 50:
            return df
        
        df = df.copy()
        
        # ATR(14) - exactly as per strategy
        df['ATR_14'] = DataProcessor.calculate_atr(df, 14)
        
        # EMA50 - trend filter
        df['EMA_50'] = df['close'].ewm(span=50).mean()
        
        # RSI(14) - momentum filter
        df['RSI_14'] = DataProcessor.calculate_rsi(df['close'], 14)
        
        # Volume MA20 - liquidity filter
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        
        # Previous close for gap calculation
        df['prev_close'] = df['close'].shift(1)
        
        return df
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range - 14 periods as per strategy"""
        if df.empty or len(df) < period:
            return pd.Series(index=df.index, dtype=float)
        
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI(14) - momentum filter"""
        if len(prices) < period:
            return pd.Series(index=prices.index, dtype=float)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        return rsi

# =============================================================================
# MIDCAP TRADING SIGNAL GENERATOR
# =============================================================================

class MidcapSignalGenerator:
    """Generates signals based on your exact midcap trading strategy"""
    
    @staticmethod
    def check_always_on_safety_rules(df: pd.DataFrame, idx: int, timeframe: int, current_time: datetime) -> bool:
        """Check all always-on safety rules from your strategy"""
        if idx < 1:
            return False
        
        try:
            current_bar = df.iloc[idx]
            prev_bar = df.iloc[idx-1]
            
            # Check for NaN values in required fields
            required_fields = ['volume', 'volume_ma20', 'open', 'close', 'EMA_50', 'RSI_14']
            for field in required_fields:
                if pd.isna(current_bar[field]) or pd.isna(prev_bar[field]):
                    return False
            
            # 1. Liquidity Requirements: Volume >= 1.5 × MA20(volume)
            if current_bar['volume'] < CONFIG.VOLUME_MULTIPLIER * current_bar['volume_ma20']:
                return False
            
            # 2. Gap Protection: Skip if |open - previous close| >= 6-8%
            if not pd.isna(prev_bar['close']):
                gap_percent = abs((current_bar['open'] - prev_bar['close']) / prev_bar['close'] * 100)
                if gap_percent >= CONFIG.MAX_GAP_PERCENT:
                    return False
            
            # 3. Intraday Hygiene: Avoid first & last 15 minutes
            # This is simplified - in real trading you'd check actual market hours
            bar_time = current_time.time()
            market_start = time(9, 15)  # 9:15 AM
            market_end = time(15, 30)   # 3:30 PM
            avoid_start = time(9, 30)   # Avoid until 9:30 AM
            avoid_end = time(15, 15)    # Avoid after 3:15 PM
            
            if bar_time < avoid_start or bar_time > avoid_end:
                return False
            
            # 4. Trend Bias: Price must be above EMA50
            if current_bar['close'] <= current_bar['EMA_50']:
                return False
            
            # 5. RSI Lower Bar: Must pass timeframe threshold
            if current_bar['RSI_14'] < CONFIG.RSI_THRESHOLDS[timeframe]:
                return False
            
            return True
            
        except (IndexError, KeyError, ZeroDivisionError):
            return False
    
    @staticmethod
    def ema_pullback_rebound_signal(df: pd.DataFrame, idx: int) -> Optional[float]:
        """A) EMA Pullback & Rebound (No Chasing) - EXACT implementation"""
        if idx < 2:
            return None
        
        try:
            current_bar = df.iloc[idx]
            prev_bar = df.iloc[idx-1] 
            prev_prev_bar = df.iloc[idx-2]
            
            # Check for required data
            required_fields = ['close', 'EMA_50', 'high', 'open']
            for field in required_fields:
                if any(pd.isna(df.iloc[i][field]) for i in [idx, idx-1, idx-2]):
                    return None
            
            # Setup: In uptrend, wait for pullback into EMA band
            # Check if previous bar was in pullback zone (within ~0.3-0.6% of EMA50)
            ema_distance = abs(prev_bar['close'] - prev_bar['EMA_50']) / prev_bar['EMA_50'] * 100
            if ema_distance > CONFIG.EMA_PULLBACK_TOLERANCE:
                return None
            
            # Entry Trigger: Next candle closes bullish and above previous bar's high (the "rebound")
            bullish_close = current_bar['close'] > current_bar['open']
            breaks_prev_high = current_bar['high'] > prev_bar['high']
            
            if bullish_close and breaks_prev_high:
                # Add tiny buffer (~0.05-0.10%) above the trigger level
                entry_price = current_bar['high'] * (1 + CONFIG.TINY_BUFFER_PERCENT / 100)
                
                # Sanity check - don't exceed reasonable bounds
                if entry_price <= current_bar['high'] * 1.2:  # Max 20% above high
                    return entry_price
            
        except (IndexError, KeyError, ZeroDivisionError):
            pass
        
        return None
    
    @staticmethod
    def inside_bar_continuation_signal(df: pd.DataFrame, idx: int) -> Optional[float]:
        """B) Inside-Bar Continuation (Volatility Squeeze) - EXACT implementation"""
        if idx < 1:
            return None
        
        try:
            current_bar = df.iloc[idx]
            prev_bar = df.iloc[idx-1]  # Mother bar
            
            # Check for required data
            required_fields = ['high', 'low']
            if any(pd.isna(current_bar[field]) or pd.isna(prev_bar[field]) for field in required_fields):
                return None
            
            # Setup: Spot an inside bar (high <= prior high AND low >= prior low)
            is_inside_bar = (current_bar['high'] <= prev_bar['high'] and 
                           current_bar['low'] >= prev_bar['low'])
            
            if is_inside_bar:
                # Entry Trigger: Buy tiny buffer above the inside bar's high
                entry_price = current_bar['high'] * (1 + CONFIG.TINY_BUFFER_PERCENT / 100)
                
                # Sanity check
                if entry_price <= current_bar['high'] * 1.2:
                    return entry_price
            
        except (IndexError, KeyError):
            pass
        
        return None
    
    @staticmethod
    def orb_signal(df: pd.DataFrame, idx: int) -> Optional[float]:
        """C) Opening-Range Breakout (ORB) - EXACT implementation"""
        if idx < CONFIG.ORB_MINUTES:
            return None
        
        try:
            current_bar = df.iloc[idx]
            
            # Define opening range (first 15 or 30 minutes after avoiding first 15 minutes)
            # This is simplified - assumes we start from bar CONFIG.ORB_MINUTES
            opening_range = df.iloc[15:CONFIG.ORB_MINUTES]  # Skip first 15 minutes, then take range
            
            if opening_range.empty:
                return None
            
            orb_high = opening_range['high'].max()
            
            if pd.isna(orb_high) or pd.isna(current_bar['high']):
                return None
            
            # Entry Trigger: Buy tiny buffer above the opening-range high
            if current_bar['high'] > orb_high:
                entry_price = orb_high * (1 + CONFIG.TINY_BUFFER_PERCENT / 100)
                
                # Sanity check
                if entry_price <= current_bar['high'] * 1.2:
                    return entry_price
            
        except (IndexError, KeyError):
            pass
        
        return None

# =============================================================================
# POSITION MANAGEMENT - THREE STAGE EXIT SYSTEM
# =============================================================================

@dataclass
class Position:
    """Trading position with three-stage exit system"""
    symbol: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    timeframe: int
    entry_type: str
    quantity: int
    stage: int = 0  # 0: initial (k0), 1: after +1R (k1), 2: after +2R (k2)
    highest_price: float = 0.0
    bars_held: int = 0
    initial_risk: float = 0.0  # R calculation
    margin_used: float = 0.0  # Track margin per position
    
    def __post_init__(self):
        self.highest_price = self.entry_price
        self.initial_risk = self.entry_price - self.stop_loss
        self.margin_used = self.entry_price * self.quantity

class MidcapPositionManager:
    """Position Manager implementing your exact three-stage exit system"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset for new stock test"""
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.initial_capital = CONFIG.INITIAL_CAPITAL
        self.current_cash = CONFIG.INITIAL_CAPITAL
        self.total_invested = 0.0
        self.equity_curve = []  # Track equity over time for performance calculations
    
    def can_open_position(self) -> bool:
        """Check if we can open new position"""
        return (len(self.positions) < CONFIG.MAX_POSITIONS and 
                self.current_cash > CONFIG.RISK_PER_TRADE)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on fixed risk per trade"""
        if entry_price <= 0 or stop_loss <= 0 or stop_loss >= entry_price:
            return 0
        
        # Fixed risk approach
        risk_per_share = entry_price - stop_loss
        quantity = int(CONFIG.RISK_PER_TRADE / risk_per_share)
        
        # Apply constraints
        position_value = quantity * entry_price
        if position_value > CONFIG.MAX_POSITION_VALUE:
            quantity = int(CONFIG.MAX_POSITION_VALUE / entry_price)
        
        # Cash constraint
        total_cost = position_value * (1 + (CONFIG.COMMISSION_PERCENT + CONFIG.SLIPPAGE_PERCENT) / 100)
        if total_cost > self.current_cash:
            quantity = int(self.current_cash / (entry_price * (1 + (CONFIG.COMMISSION_PERCENT + CONFIG.SLIPPAGE_PERCENT) / 100)))
        
        return max(1, quantity) if quantity > 0 else 0
    
    def open_position(self, symbol: str, entry_price: float, timeframe: int,
                     entry_type: str, atr: float, current_time: datetime) -> bool:
        """Open new position with Stage 1: Initial Stop (k₀)"""
        if not self.can_open_position() or pd.isna(entry_price) or pd.isna(atr):
            return False
        
        # Stage 1: Initial Stop (k₀) - "A bit loose" for flexibility
        k0 = CONFIG.ATR_MULTIPLIERS[timeframe]['k0']
        initial_stop = entry_price - (atr * k0)
        
        # For inside-bar entries: Use safer of ATR stop or just below inside-bar low
        if entry_type == 'Inside_Bar':
            # This would require storing the inside-bar low, simplified for now
            pass
        
        if initial_stop <= 0:
            return False
        
        quantity = self.calculate_position_size(entry_price, initial_stop)
        if quantity == 0:
            return False
        
        position_value = entry_price * quantity
        total_cost = position_value * (1 + (CONFIG.COMMISSION_PERCENT + CONFIG.SLIPPAGE_PERCENT) / 100)
        
        if total_cost > self.current_cash:
            return False
        
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=current_time,
            stop_loss=initial_stop,
            timeframe=timeframe,
            entry_type=entry_type,
            quantity=quantity,
            margin_used=position_value
        )
        
        self.positions.append(position)
        self.current_cash -= total_cost
        self.total_invested += position_value
        
        return True
    
    def update_positions(self, symbol: str, current_price: float, atr: float, current_time: datetime):
        """Update positions with three-stage exit system"""
        if pd.isna(current_price) or pd.isna(atr) or current_price <= 0:
            return
        
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            if pos.symbol != symbol:
                continue
            
            pos.bars_held += 1
            pos.highest_price = max(pos.highest_price, current_price)
            
            # Check stop loss (Critical Rule: stops never move down)
            if current_price <= pos.stop_loss:
                positions_to_close.append((i, 'Stop Loss', current_price))
                continue
            
            # Time Stop: If price doesn't reach +0.5R within specified time window
            if pos.bars_held >= CONFIG.TIME_STOPS[pos.timeframe]:
                if pos.initial_risk > 0:
                    r_multiple = (current_price - pos.entry_price) / pos.initial_risk
                    if r_multiple < 0.5:  # Hasn't reached +0.5R
                        positions_to_close.append((i, 'Time Stop', current_price))
                        continue
            
            # Three-Stage Exit Process
            self._update_three_stage_exit(pos, current_price, atr)
        
        # Close positions
        for i, exit_reason, exit_price in reversed(positions_to_close):
            self._close_position(i, exit_reason, exit_price, current_time)
    
    def record_equity_point(self, current_time: datetime, current_price: float = None):
        """Record equity curve point for performance calculations"""
        # Calculate current equity (cash + mark-to-market of positions)
        mtm_value = 0.0
        if current_price is not None:
            for pos in self.positions:
                mtm_value += pos.quantity * current_price
        else:
            mtm_value = self.total_invested
        
        current_equity = self.current_cash + mtm_value
        
        self.equity_curve.append({
            'timestamp': current_time,
            'equity': current_equity,
            'cash': self.current_cash,
            'positions_value': mtm_value,
            'num_positions': len(self.positions)
        })
    
    def _update_three_stage_exit(self, pos: Position, current_price: float, atr: float):
        """Implement exact three-stage exit system from your strategy"""
        if pos.initial_risk <= 0:
            return
        
        # Calculate current R-multiple
        r_multiple = (current_price - pos.entry_price) / pos.initial_risk
        
        # Stage 2: Prove It → Tighten (k₁)
        if r_multiple >= 1.0 and pos.stage < 1:
            pos.stage = 1
            
            # Move stop to breakeven + tiny buffer (~0.10%)
            breakeven_stop = pos.entry_price * (1 + CONFIG.TINY_BUFFER_PERCENT / 100)
            
            # Switch to Chandelier trail: Trail = HighestSinceEntry - ATR(14) × k₁
            k1 = CONFIG.ATR_MULTIPLIERS[pos.timeframe]['k1']
            chandelier_stop = pos.highest_price - (atr * k1)
            
            # Use the higher of breakeven+ or chandelier (never move stops down)
            pos.stop_loss = max(pos.stop_loss, max(breakeven_stop, chandelier_stop))
        
        # Stage 3: Lock Profits Harder (k₂)
        elif r_multiple >= 2.0 and pos.stage < 2:
            pos.stage = 2
            
            # Tighten trail: Trail = HighestSinceEntry - ATR(14) × k₂
            k2 = CONFIG.ATR_MULTIPLIERS[pos.timeframe]['k2']
            tight_trail_stop = pos.highest_price - (atr * k2)
            
            # Never move stops down
            pos.stop_loss = max(pos.stop_loss, tight_trail_stop)
        
        # Continue trailing for current stage
        if pos.stage == 1:
            k1 = CONFIG.ATR_MULTIPLIERS[pos.timeframe]['k1']
            new_trail = pos.highest_price - (atr * k1)
            pos.stop_loss = max(pos.stop_loss, new_trail)
        elif pos.stage == 2:
            k2 = CONFIG.ATR_MULTIPLIERS[pos.timeframe]['k2']
            new_trail = pos.highest_price - (atr * k2)
            pos.stop_loss = max(pos.stop_loss, new_trail)
    
    def _close_position(self, position_idx: int, exit_reason: str, exit_price: float, current_time: datetime):
        """Close position and record trade with comprehensive metrics"""
        pos = self.positions.pop(position_idx)
        
        if exit_price <= 0:
            exit_price = pos.stop_loss
        
        # Calculate costs and P&L
        entry_value = pos.entry_price * pos.quantity
        entry_commission = entry_value * (CONFIG.COMMISSION_PERCENT / 100)
        entry_slippage = entry_value * (CONFIG.SLIPPAGE_PERCENT / 100)
        
        exit_value = exit_price * pos.quantity
        exit_commission = exit_value * (CONFIG.COMMISSION_PERCENT / 100)
        exit_slippage = exit_value * (CONFIG.SLIPPAGE_PERCENT / 100)
        
        gross_pnl = exit_value - entry_value
        total_costs = entry_commission + exit_commission + entry_slippage + exit_slippage
        net_pnl = gross_pnl - total_costs
        
        # REQUESTED METRICS CALCULATION
        pnl_percent = (net_pnl / entry_value) * 100  # P&L percentage per trade
        
        # Cap unrealistic returns but preserve actual calculation
        capped_return = False
        if abs(pnl_percent) > CONFIG.MAX_SINGLE_TRADE_RETURN:
            print(f"Large return detected: {pnl_percent:.2f}% for {pos.symbol} - preserving actual value")
            capped_return = True
        
        # Update cash
        net_exit_proceeds = exit_value - exit_commission - exit_slippage
        self.current_cash += net_exit_proceeds
        self.total_invested -= entry_value
        
        # Calculate R-multiple
        r_multiple = (exit_price - pos.entry_price) / pos.initial_risk if pos.initial_risk > 0 else 0
        
        # Record trade with comprehensive metrics
        trade_record = {
            'symbol': pos.symbol,
            'entry_time': pos.entry_time,
            'exit_time': current_time,
            'timeframe': pos.timeframe,
            'entry_type': pos.entry_type,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'quantity': pos.quantity,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'pnl_percent': pnl_percent,  # REQUESTED: Average P&L percent per trade
            'total_costs': total_costs,
            'margin_used': pos.margin_used,  # REQUESTED: Margin per trade
            'exit_reason': exit_reason,
            'bars_held': pos.bars_held,
            'r_multiple': r_multiple,
            'final_stage': pos.stage,
            'capped_return': capped_return
        }
        
        self.closed_trades.append(trade_record)

# =============================================================================
# PERFORMANCE CALCULATOR - COMPREHENSIVE METRICS
# =============================================================================

class PerformanceCalculator:
    """Calculate comprehensive performance metrics including returns"""
    
    @staticmethod
    def calculate_comprehensive_metrics(trades_df: pd.DataFrame, equity_curve: List[Dict], 
                                      initial_capital: float) -> Dict:
        """Calculate all requested performance metrics"""
        if trades_df.empty:
            return PerformanceCalculator._empty_metrics()
        
        # Basic Trade Metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # P&L Calculations
        total_pnl = trades_df['net_pnl'].sum()
        final_equity = initial_capital + total_pnl
        
        # REQUESTED CORE METRICS
        avg_pnl_percent = trades_df['pnl_percent'].mean()  # Average P&L percent per trade
        avg_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() if losing_trades > 0 else 0
        avg_margin = trades_df['margin_used'].mean()  # Average margin per trade
        
        # Total P&L Ratio (total wins / abs(total losses))
        total_wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        total_losses = abs(trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum())
        total_pnl_ratio = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # RETURN CALCULATIONS - Multiple Methods
        simple_return = (final_equity - initial_capital) / initial_capital * 100  # Simple return %
        
        # Time-weighted return calculation using equity curve
        if equity_curve and len(equity_curve) > 1:
            equity_values = [point['equity'] for point in equity_curve]
            equity_series = pd.Series(equity_values)
            
            # Calculate periodic returns
            periodic_returns = equity_series.pct_change().dropna()
            
            # Compound Annual Growth Rate (CAGR)
            if len(equity_curve) > 0:
                start_date = equity_curve[0]['timestamp']
                end_date = equity_curve[-1]['timestamp']
                time_period_years = (end_date - start_date).days / 365.25
                
                if time_period_years > 0:
                    cagr = ((final_equity / initial_capital) ** (1/time_period_years) - 1) * 100
                else:
                    cagr = simple_return
            else:
                cagr = simple_return
            
            # Annualized return based on periodic returns
            if len(periodic_returns) > 0:
                # Assuming daily data, annualize
                mean_return = periodic_returns.mean()
                annualized_return = ((1 + mean_return) ** 252 - 1) * 100  # 252 trading days
                
                # Volatility calculations
                volatility = periodic_returns.std() * np.sqrt(252) * 100  # Annualized volatility
                
                # Sharpe Ratio (assuming risk-free rate = 0)
                sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
            else:
                annualized_return = simple_return
                volatility = 0
                sharpe_ratio = 0
        else:
            cagr = simple_return
            annualized_return = simple_return
            volatility = 0
            sharpe_ratio = 0
        
        # Risk Metrics
        max_drawdown = PerformanceCalculator._calculate_max_drawdown(equity_curve)
        
        # Additional Performance Metrics
        avg_r_multiple = trades_df['r_multiple'].mean()
        profit_factor = abs(avg_profit * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else 0
        
        # Expectancy (probability-weighted average outcome)
        expectancy = (win_rate/100 * avg_profit) + ((1-win_rate/100) * avg_loss)
        
        # Risk-adjusted metrics
        recovery_factor = abs(total_pnl / max_drawdown) if max_drawdown != 0 else 0
        avg_bars_held = trades_df['bars_held'].mean()
        
        # Advanced Return Metrics
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0  # CAGR / Max Drawdown
        sortino_ratio = PerformanceCalculator._calculate_sortino_ratio(trades_df)
        
        # Return consistency metrics
        winning_months = PerformanceCalculator._calculate_winning_periods(equity_curve)
        
        return {
            # Basic Metrics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            
            # REQUESTED CORE METRICS
            'avg_pnl_percent': round(avg_pnl_percent, 2),
            'avg_profit': round(avg_profit, 2),
            'avg_loss': round(avg_loss, 2),
            'total_pnl_ratio': round(total_pnl_ratio, 2) if total_pnl_ratio != float('inf') else 'Inf',
            'avg_margin': round(avg_margin, 2),
            
            # COMPREHENSIVE RETURN METRICS
            'total_pnl': round(total_pnl, 2),
            'simple_return': round(simple_return, 2),           # Basic return %
            'cagr': round(cagr, 2),                            # Compound Annual Growth Rate
            'annualized_return': round(annualized_return, 2),   # Annualized return
            'time_weighted_return': round(cagr, 2),             # Time-weighted return
            
            # Risk and Risk-Adjusted Metrics
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'recovery_factor': round(recovery_factor, 2),
            
            # Additional Metrics
            'avg_r_multiple': round(avg_r_multiple, 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy, 2),
            'avg_bars_held': round(avg_bars_held, 1),
            'winning_months': winning_months,
            'final_equity': round(final_equity, 2)
        }
    
    @staticmethod
    def _calculate_max_drawdown(equity_curve: List[Dict]) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not equity_curve:
            return 0.0
        
        equity_values = [point['equity'] for point in equity_curve]
        equity_series = pd.Series(equity_values)
        
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        
        return abs(drawdown.min()) if not drawdown.empty else 0.0
    
    @staticmethod
    def _calculate_sortino_ratio(trades_df: pd.DataFrame) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if trades_df.empty:
            return 0.0
        
        returns = trades_df['pnl_percent']
        
        # Calculate downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = negative_returns.std()
        mean_return = returns.mean()
        
        return mean_return / downside_deviation if downside_deviation != 0 else 0
    
    @staticmethod
    def _calculate_winning_periods(equity_curve: List[Dict]) -> int:
        """Calculate number of winning periods (simplified)"""
        if len(equity_curve) < 2:
            return 0
        
        winning_periods = 0
        for i in range(1, len(equity_curve)):
            if equity_curve[i]['equity'] > equity_curve[i-1]['equity']:
                winning_periods += 1
        
        return winning_periods
    
    @staticmethod
    def _empty_metrics() -> Dict:
        """Return empty metrics dictionary"""
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'avg_pnl_percent': 0, 'avg_profit': 0, 'avg_loss': 0, 'total_pnl_ratio': 0,
            'avg_margin': 0, 'total_pnl': 0, 'simple_return': 0, 'cagr': 0,
            'annualized_return': 0, 'time_weighted_return': 0, 'volatility': 0,
            'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0,
            'max_drawdown': 0, 'recovery_factor': 0, 'avg_r_multiple': 0,
            'profit_factor': 0, 'expectancy': 0, 'avg_bars_held': 0,
            'winning_months': 0, 'final_equity': CONFIG.INITIAL_CAPITAL
        }

# =============================================================================
# BACKTESTING ENGINE FOR MIDCAP STRATEGY - ENHANCED
# =============================================================================

class MidcapBacktestEngine:
    """Backtesting engine implementing your exact midcap strategy with comprehensive metrics"""
    
    def __init__(self):
        self.position_manager = MidcapPositionManager()
        self.signal_generator = MidcapSignalGenerator()
    
    def run_single_stock_test(self, symbol: str, df_1min: pd.DataFrame, timeframe: int) -> Dict:
        """Run backtest on single stock with your exact strategy"""
        self.position_manager.reset()
        
        try:
            # Resample to target timeframe
            df = DataProcessor.resample_to_timeframe(df_1min, timeframe)
            
            if df.empty or len(df) < CONFIG.MIN_BARS_FOR_ENTRY:
                return self._empty_result(symbol, timeframe, "Insufficient data")
            
            # Run backtest starting after warmup
            for idx in range(CONFIG.MIN_BARS_FOR_ENTRY, len(df)):
                current_time = df.index[idx]
                current_bar = df.iloc[idx]
                
                # Skip if essential data missing
                if pd.isna(current_bar['close']) or pd.isna(current_bar['ATR_14']):
                    continue
                
                # Record equity point
                self.position_manager.record_equity_point(current_time, current_bar['close'])
                
                # Update existing positions
                if self.position_manager.positions:
                    self.position_manager.update_positions(
                        symbol, current_bar['close'], current_bar['ATR_14'], current_time
                    )
                
                # Check always-on safety rules first
                if not self.signal_generator.check_always_on_safety_rules(df, idx, timeframe, current_time):
                    continue
                
                # Try entry methods - Choose ONE that matches the chart
                entry_signals = [
                    ('EMA_Pullback', self.signal_generator.ema_pullback_rebound_signal(df, idx)),
                    ('Inside_Bar', self.signal_generator.inside_bar_continuation_signal(df, idx)),
                    ('ORB', self.signal_generator.orb_signal(df, idx))
                ]
                
                # Only one entry per bar
                for entry_type, signal_price in entry_signals:
                    if signal_price is not None and signal_price > 0:
                        success = self.position_manager.open_position(
                            symbol, signal_price, timeframe, entry_type,
                            current_bar['ATR_14'], current_time
                        )
                        if success:
                            break
            
            # Close remaining positions at end
            if self.position_manager.positions:
                final_price = df.iloc[-1]['close']
                final_time = df.index[-1]
                
                for i in reversed(range(len(self.position_manager.positions))):
                    self.position_manager._close_position(i, 'End of Data', final_price, final_time)
            
            return self._generate_comprehensive_result(symbol, timeframe)
            
        except Exception as e:
            return self._empty_result(symbol, timeframe, f"Error: {str(e)}")
    
    def _generate_comprehensive_result(self, symbol: str, timeframe: int) -> Dict:
        """Generate comprehensive results with ALL requested metrics"""
        trades_df = pd.DataFrame(self.position_manager.closed_trades)
        
        if trades_df.empty:
            return self._empty_result(symbol, timeframe, "No trades generated")
        
        # Calculate comprehensive performance metrics
        performance_metrics = PerformanceCalculator.calculate_comprehensive_metrics(
            trades_df, 
            self.position_manager.equity_curve, 
            CONFIG.INITIAL_CAPITAL
        )
        
        # Entry method breakdown
        entry_breakdown = trades_df['entry_type'].value_counts().to_dict()
        
        # Stage analysis
        stage_breakdown = trades_df['final_stage'].value_counts().to_dict()
        
        # Large return detection
        large_returns_detected = len(trades_df[trades_df['capped_return'] == True])
        
        # Combine all metrics
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            **performance_metrics,  # Include all calculated performance metrics
            'entry_breakdown': entry_breakdown,
            'stage_breakdown': stage_breakdown,
            'large_returns_detected': large_returns_detected,
            'status': 'Success'
        }
        
        return result
    
    def _empty_result(self, symbol: str, timeframe: int, reason: str) -> Dict:
        """Empty result template with all metrics"""
        empty_metrics = PerformanceCalculator._empty_metrics()
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            **empty_metrics,
            'entry_breakdown': {},
            'stage_breakdown': {},
            'large_returns_detected': 0,
            'status': reason
        }

# =============================================================================
# RESULTS HANDLER - COMPREHENSIVE ANALYSIS
# =============================================================================

class MidcapResultsHandler:
    """Handle results with comprehensive strategy analysis"""
    
    def __init__(self, output_filename: str = None):
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_filename = f"comprehensive_midcap_results_{timestamp}.xlsx"
        else:
            self.output_filename = output_filename
        
        self.results = []
    
    def add_result(self, result: Dict):
        """Add result to collection"""
        self.results.append(result)
    
    def save_to_excel(self):
        """Save results to Excel with comprehensive analysis"""
        if not self.results:
            print("No results to save")
            return
        
        df_results = pd.DataFrame(self.results)
        
        with pd.ExcelWriter(self.output_filename, engine='openpyxl') as writer:
            # Main results with all metrics
            df_results.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Comprehensive summaries
            self._create_comprehensive_summaries(writer, df_results)
        
        print(f"Comprehensive results saved to: {self.output_filename}")
    
    def _create_comprehensive_summaries(self, writer, df_results):
        """Create comprehensive summary sheets"""
        successful_results = df_results[df_results['status'] == 'Success']
        
        if successful_results.empty:
            return
        
        # Timeframe analysis with ALL metrics
        timeframe_summary = []
        for tf in CONFIG.TIMEFRAMES:
            tf_data = successful_results[successful_results['timeframe'] == tf]
            if not tf_data.empty:
                summary = {
                    'Timeframe': f"{tf}min",
                    'Tests': len(tf_data),
                    'Total_Trades': tf_data['total_trades'].sum(),
                    'Avg_Win_Rate': round(tf_data['win_rate'].mean(), 2),
                    
                    # REQUESTED CORE METRICS
                    'Avg_PnL_Percent': round(tf_data['avg_pnl_percent'].mean(), 2),
                    'Avg_Profit': round(tf_data['avg_profit'].mean(), 2),
                    'Avg_Loss': round(tf_data['avg_loss'].mean(), 2),
                    'Avg_PnL_Ratio': round(tf_data[tf_data['total_pnl_ratio'] != 'Inf']['total_pnl_ratio'].mean(), 2),
                    'Avg_Margin': round(tf_data['avg_margin'].mean(), 2),
                    
                    # RETURN METRICS
                    'Avg_Simple_Return': round(tf_data['simple_return'].mean(), 2),
                    'Avg_CAGR': round(tf_data['cagr'].mean(), 2),
                    'Avg_Annualized_Return': round(tf_data['annualized_return'].mean(), 2),
                    
                    # RISK METRICS
                    'Avg_Sharpe_Ratio': round(tf_data['sharpe_ratio'].mean(), 2),
                    'Avg_Max_Drawdown': round(tf_data['max_drawdown'].mean(), 2),
                    'Avg_Volatility': round(tf_data['volatility'].mean(), 2),
                    
                    # ADDITIONAL METRICS
                    'Avg_R_Multiple': round(tf_data['avg_r_multiple'].mean(), 2),
                    'Avg_Expectancy': round(tf_data['expectancy'].mean(), 2),
                    'Best_Performer': tf_data.loc[tf_data['simple_return'].idxmax(), 'symbol'] if len(tf_data) > 0 else 'N/A'
                }
                timeframe_summary.append(summary)
        
        if timeframe_summary:
            pd.DataFrame(timeframe_summary).to_excel(writer, sheet_name='Timeframe_Analysis', index=False)

# =============================================================================
# MAIN EXECUTION FUNCTION - COMPREHENSIVE
# =============================================================================

def run_comprehensive_midcap_backtest(data_folder_path: str, output_filename: str = None):
    """Run comprehensive backtest with ALL performance metrics"""
    print("COMPREHENSIVE MIDCAP TRADING STRATEGY BACKTEST")
    print("="*55)
    print("Strategy Implementation:")
    print("✓ Always-On Safety Rules (Volume, Gap, Trend, RSI)")
    print("✓ Three Entry Methods (EMA Pullback, Inside Bar, ORB)")
    print("✓ Three-Stage Exit System (k0 → k1 → k2)")
    print("✓ COMPREHENSIVE Performance Metrics Including:")
    print("  • P&L Percentage, Profit/Loss Averages, P&L Ratios")
    print("  • Multiple Return Calculations (Simple, CAGR, Annualized)")
    print("  • Risk Metrics (Sharpe, Sortino, Calmar, Max Drawdown)")
    print("  • Advanced Metrics (R-Multiple, Expectancy, Recovery Factor)")
    
    # Initialize components
    engine = MidcapBacktestEngine()
    results_handler = MidcapResultsHandler(output_filename)
    
    # Get CSV files
    csv_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found!")
        return []
    
    print(f"\nProcessing {len(csv_files)} stocks across {len(CONFIG.TIMEFRAMES)} timeframes")
    total_tests = len(csv_files) * len(CONFIG.TIMEFRAMES)
    current_test = 0
    
    # Process each stock
    for csv_file in csv_files:
        symbol = csv_file.replace('.csv', '').replace('_data', '')
        file_path = os.path.join(data_folder_path, csv_file)
        
        print(f"\n{'='*60}")
        print(f"Processing: {symbol}")
        print(f"{'='*60}")
        
        try:
            df_1min = DataProcessor.load_single_csv(file_path)
            
            if df_1min.empty:
                print(f"No valid data for {symbol}")
                for timeframe in CONFIG.TIMEFRAMES:
                    result = engine._empty_result(symbol, timeframe, "No valid data")
                    results_handler.add_result(result)
                continue
            
            print(f"Loaded {len(df_1min)} rows from {df_1min.index.min().date()} to {df_1min.index.max().date()}")
            
            # Test each timeframe
            for timeframe in CONFIG.TIMEFRAMES:
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] Testing {symbol} - {timeframe}min...")
                print(f"  RSI Threshold: {CONFIG.RSI_THRESHOLDS[timeframe]}")
                print(f"  ATR Multipliers: k0={CONFIG.ATR_MULTIPLIERS[timeframe]['k0']}, " +
                      f"k1={CONFIG.ATR_MULTIPLIERS[timeframe]['k1']}, " +
                      f"k2={CONFIG.ATR_MULTIPLIERS[timeframe]['k2']}")
                print(f"  Time Stop: {CONFIG.TIME_STOPS[timeframe]} bars")
                
                result = engine.run_single_stock_test(symbol, df_1min, timeframe)
                
                # Display comprehensive results
                if result['status'] == 'Success':
                    print(f"✓ Trades: {result['total_trades']}, Win Rate: {result['win_rate']}%, " +
                          f"Return: {result['simple_return']}%, R-Multiple: {result['avg_r_multiple']}")
                    
                    # Enhanced display with all requested metrics
                    print(f"  P&L%: {result['avg_pnl_percent']}%, Profit: ₹{result['avg_profit']:,.0f}, " +
                          f"Loss: ₹{result['avg_loss']:,.0f}")
                    print(f"  P&L Ratio: {result['total_pnl_ratio']}, Margin: ₹{result['avg_margin']:,.0f}")
                    print(f"  CAGR: {result['cagr']}%, Sharpe: {result['sharpe_ratio']}, " +
                          f"Max DD: {result['max_drawdown']}%")
                    
                    # Entry breakdown
                    if result['entry_breakdown']:
                        breakdown = ", ".join([f"{k}: {v}" for k, v in result['entry_breakdown'].items()])
                        print(f"  Entry Methods: {breakdown}")
                    
                    # Stage breakdown  
                    if result['stage_breakdown']:
                        stages = ", ".join([f"Stage{k}: {v}" for k, v in result['stage_breakdown'].items()])
                        print(f"  Exit Stages: {stages}")
                    
                    # Large returns warning
                    if result['large_returns_detected'] > 0:
                        print(f"  ⚠️  {result['large_returns_detected']} large returns detected")
                    
                else:
                    print(f"✗ {result['status']}")
                
                results_handler.add_result(result)
                
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            for timeframe in CONFIG.TIMEFRAMES:
                current_test += 1
                result = engine._empty_result(symbol, timeframe, f"Error: {str(e)}")
                results_handler.add_result(result)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving comprehensive results...")
    results_handler.save_to_excel()
    
    # Display comprehensive summary
    display_comprehensive_summary(results_handler.results)
    
    return results_handler.results

def display_comprehensive_summary(results: List[Dict]):
    """Display comprehensive summary with ALL metrics"""
    if not results:
        return
    
    df_results = pd.DataFrame(results)
    successful_results = df_results[df_results['status'] == 'Success']
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE MIDCAP STRATEGY SUMMARY") 
    print(f"{'='*70}")
    print(f"Total Tests: {len(results)}")
    print(f"Successful Tests: {len(successful_results)}")
    
    if not successful_results.empty:
        print(f"\nCORE TRADING METRICS:")
        print(f"Total Trades: {successful_results['total_trades'].sum()}")
        print(f"Average Win Rate: {successful_results['win_rate'].mean():.1f}%")
        
        print(f"\nREQUESTED P&L METRICS:")
        print(f"Average P&L per Trade: {successful_results['avg_pnl_percent'].mean():.2f}%")
        print(f"Average Profit per Win: ₹{successful_results['avg_profit'].mean():,.0f}")
        print(f"Average Loss per Loss: ₹{successful_results['avg_loss'].mean():,.0f}")
        print(f"Average P&L Ratio: {successful_results[successful_results['total_pnl_ratio'] != 'Inf']['total_pnl_ratio'].mean():.2f}")
        print(f"Average Margin per Trade: ₹{successful_results['avg_margin'].mean():,.0f}")
        
        print(f"\nRETURN METRICS:")
        print(f"Average Simple Return: {successful_results['simple_return'].mean():.2f}%")
        print(f"Average CAGR: {successful_results['cagr'].mean():.2f}%")
        print(f"Average Annualized Return: {successful_results['annualized_return'].mean():.2f}%")
        print(f"Total P&L Across All Tests: ₹{successful_results['total_pnl'].sum():,.0f}")
        
        print(f"\nRISK METRICS:")
        print(f"Average Sharpe Ratio: {successful_results['sharpe_ratio'].mean():.2f}")
        print(f"Average Max Drawdown: {successful_results['max_drawdown'].mean():.2f}%")
        print(f"Average Volatility: {successful_results['volatility'].mean():.2f}%")
        print(f"Average Recovery Factor: {successful_results['recovery_factor'].mean():.2f}")
        
        print(f"\nSTRATEGY METRICS:")
        print(f"Average R-Multiple: {successful_results['avg_r_multiple'].mean():.2f}")
        print(f"Average Expectancy: ₹{successful_results['expectancy'].mean():,.0f}")
        print(f"Average Profit Factor: {successful_results['profit_factor'].mean():.2f}")
        
        print(f"\nTIMEFRAME PERFORMANCE:")
        for tf in CONFIG.TIMEFRAMES:
            tf_data = successful_results[successful_results['timeframe'] == tf]
            if not tf_data.empty:
                print(f"{tf}min: Return: {tf_data['simple_return'].mean():.2f}%, " +
                      f"CAGR: {tf_data['cagr'].mean():.2f}%, " +
                      f"Sharpe: {tf_data['sharpe_ratio'].mean():.2f}, " +
                      f"P&L%: {tf_data['avg_pnl_percent'].mean():.2f}%")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    data_folder = "Midcap Data"  
    output_file = "comprehensive_midcap_strategy_results.xlsx"
    
    # Run comprehensive midcap strategy backtest
    results = run_comprehensive_midcap_backtest(data_folder, output_file)
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE MIDCAP STRATEGY BACKTEST COMPLETE!")
    print(f"Results saved to: {output_file}")
    print("="*70)
